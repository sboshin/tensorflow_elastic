#include "tensorflow/core/kernels/data/dataset_utils.h"

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow
{
  namespace data
  {
    namespace
    {
      constexpr char kDelimiter[] = "@@";
      constexpr char kComponent[] = "component";
      constexpr char kNumElements[] = "num_elements";
      constexpr char kNumComponents[] = "num_components";
    } // namespace

    namespace
    {

      // We assume that all keys are of the form <iterator_prefix>:<name>. We extract
      // the iterator name by getting rid of everything post the final colon.
      Status GetIteratorName(StringPiece key, string *name)
      {
        if (!str_util::StartsWith(key, data::kFullNameRandomHex))
        {
          return errors::InvalidArgument("Save key: ", key,
                                         " not generated using full_name.");
        }
        std::vector<string> split_keys = str_util::Split(key, data::kPipe);
        if (split_keys.size() != 2)
        {
          return errors::InvalidArgument("Save key: ", key,
                                         " not generated using full_name.");
        }
        string real_key = split_keys[1];
        const int pos = real_key.rfind(kColon);
        *name = real_key.substr(0, pos);
        return Status::OK();
      }

    } // namespace

    // Helper class used to build a list of VariantTensorData objects, one for each
    // iterator which is determined from the key supplied from the Write* calls.
    // Sample usage:
    // EVariantTensorDataWriter writer;
    // writer.WriteScalar(full_name("buffer_size"), buffer_.size());
    // writer.WriteScalar(full_name("num_threads"), threadpool_.size());
    // ....
    // std::vector<std::unique_ptr<VariantTensorData>> variants;
    // writer.ReleaseData(&variants);
    // Now the VariantTensorData objects can be used to serialize.
    class EVariantTensorDataWriter : public VariantTensorDataWriter
    {
    public:
      Status WriteScalar(StringPiece key, const int64 val) override;
      Status WriteScalar(StringPiece key, const tstring &val) override;
      Status WriteTensor(StringPiece key, const Tensor &val) override;

      Status WriteScalar(StringPiece name, StringPiece key,
                         const int64 val) override;
      Status WriteScalar(StringPiece name, StringPiece key,
                         const tstring &val) override;
      Status WriteTensor(StringPiece name, StringPiece key,
                         const Tensor &val) override;

      // Releases the built VariantTensorData's to `variants`. Clears out all
      // class state.
      void ReleaseData(std::vector<std::unique_ptr<VariantTensorData>> *variants);

      // Obtains a read-only version of the VariantTensorData's built.
      void GetData(std::vector<const VariantTensorData *> *variants);

    private:
      void MaybeFlush();
      void Reset();

      template <typename T>
      Status WriteScalarInternal(StringPiece key, const T &val);
      Status WriteTensorInternal(StringPiece key, const Tensor &val);

      template <typename T>
      Status WriteScalarInternal(StringPiece name, StringPiece key, const T &val);
      Status WriteTensorInternal(StringPiece name, StringPiece key,
                                 const Tensor &val);

      bool is_flushed_ = false;
      std::map<string, std::unique_ptr<VariantTensorData>> data_;
      std::map<string, std::vector<string>> keys_;
    };

    Status
    EVariantTensorDataWriter::WriteScalar(StringPiece key, const int64 val)
    {
      return WriteScalarInternal(key, val);
    }

    Status EVariantTensorDataWriter::WriteScalar(StringPiece key,
                                                 const tstring &val)
    {
      return WriteScalarInternal(key, val);
    }

    Status EVariantTensorDataWriter::WriteTensor(StringPiece key,
                                                 const Tensor &val)
    {
      return WriteTensorInternal(key, val);
    }

    Status EVariantTensorDataWriter::WriteScalar(StringPiece name, StringPiece key,
                                                 const int64 val)
    {
      return WriteScalarInternal(name, key, val);
    }

    Status EVariantTensorDataWriter::WriteScalar(StringPiece name, StringPiece key,
                                                 const tstring &val)
    {
      return WriteScalarInternal(name, key, val);
    }

    Status EVariantTensorDataWriter::WriteTensor(StringPiece name, StringPiece key,
                                                 const Tensor &val)
    {
      return WriteTensorInternal(name, key, val);
    }

    void EVariantTensorDataWriter::MaybeFlush()
    {
      if (is_flushed_)
        return;
      for (auto &keys : keys_)
      {
        const string name = keys.first;
        string metadata = name;
        for (size_t i = 0; i < keys_[name].size(); ++i)
        {
          strings::StrAppend(&metadata, kDelimiter, keys_[name][i]);
        }
        data_[name]->set_metadata(metadata);
      }
      is_flushed_ = true;
    }

    void EVariantTensorDataWriter::Reset()
    {
      is_flushed_ = false;
      data_.clear();
      keys_.clear();
    }

    void EVariantTensorDataWriter::ReleaseData(
        std::vector<std::unique_ptr<VariantTensorData>> *variants)
    {
      MaybeFlush();
      for (auto &it : data_)
      {
        variants->push_back(std::move(it.second));
      }
      Reset();
    }

    void EVariantTensorDataWriter::GetData(
        std::vector<const VariantTensorData *> *variants)
    {
      MaybeFlush();
      for (auto &it : data_)
      {
        variants->push_back(it.second.get());
      }
    }

    template <typename T>
    Status EVariantTensorDataWriter::WriteScalarInternal(StringPiece key,
                                                         const T &val)
    {
      if (is_flushed_)
      {
        return errors::FailedPrecondition(
            "Cannot call WriteScalar after GetData or ReleaseData is called");
      }
      string name;
      TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
      return WriteScalarInternal(name, key, val);
    }

    Status EVariantTensorDataWriter::WriteTensorInternal(StringPiece key,
                                                         const Tensor &val)
    {
      if (is_flushed_)
      {
        return errors::FailedPrecondition(
            "Cannot call WriteTensor after GetData or ReleaseData is called");
      }
      string name;
      TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
      return WriteTensorInternal(name, key, val);
    }

    template <typename T>
    Status EVariantTensorDataWriter::WriteScalarInternal(StringPiece name,
                                                         StringPiece key,
                                                         const T &val)
    {
      if (is_flushed_)
      {
        return errors::FailedPrecondition(
            "Cannot call WriteScalar after GetData or ReleaseData is called");
      }
      Tensor val_t = Tensor(DataTypeToEnum<T>::v(), TensorShape({}));
      val_t.scalar<T>()() = val;
      return WriteTensorInternal(name, key, val_t);
    }

    Status EVariantTensorDataWriter::WriteTensorInternal(StringPiece n,
                                                         StringPiece key,
                                                         const Tensor &val)
    {
      if (is_flushed_)
      {
        return errors::FailedPrecondition(
            "Cannot call WriteTensor after GetData or ReleaseData is called");
      }
      DCHECK_EQ(key.find(kDelimiter), string::npos);
      string name(n);
      if (keys_.count(name) == 0)
      {
        keys_[name] = std::vector<string>();
      }
      keys_[name].push_back(string(key));
      if (data_.count(name) == 0)
      {
        data_[name] = absl::make_unique<VariantTensorData>();
        data_[name]->set_type_name("tensorflow::EIterator");
      }
      *(data_[name]->add_tensors()) = val;
      return Status::OK();
    }

  } // namespace data
} // namespace tensorflow
