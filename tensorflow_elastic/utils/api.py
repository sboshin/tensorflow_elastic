#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket

used_ports = []

def get_env_variable_or_raise(env_name: str) -> str:
    r"""
    Tries to retrieve environment variable. Raises ``ValueError``
    if no environment variable found.

    Arguments:
        env_name (str): Name of the env variable
    """
    value = os.environ.get(env_name, None)
    if value is None:
        msg = f"Environment variable {env_name} expected, but not set"
        raise ValueError(msg)
    return value

def PickUnusedPort():
  s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
  s.bind(('', 0))
  port = s.getsockname()[1]
  s.close()
  return port

def get_socket_with_port():
  return socket.gethostname()+":"+str(PickUnusedPort())