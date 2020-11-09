requirements:
    grpcio
    grpcio-tools [ need to run protoc ]

run the following command to re-generate protobuf stub code for python:

python -m grpc_tools.protoc  --python_out=. --grpc_python_out=. -I. hypernets/dispatchers/cluster/grpc/proto/spec.proto
