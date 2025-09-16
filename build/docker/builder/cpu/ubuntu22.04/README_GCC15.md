# GCC 15 升级说明

## 升级内容

本次升级将 Milvus Ubuntu 22.04 构建环境从 GCC 12 升级到 GCC 15，主要变更包括：

### 编译器升级
- **GCC**: 12 → 15.1.0 (从源码编译)
- **CMake**: 3.31.8 → 3.32.0
- 保持 Go 1.24.4 和 Conan 1.64.1 不变

### 关键变更
1. **GCC 15 源码编译**: 由于 Ubuntu 22.04 软件源不提供 GCC 15，采用源码编译方式
2. **依赖库更新**: 安装 GCC 编译所需的依赖库 (libgmp-dev, libmpfr-dev, libmpc-dev)
3. **环境变量设置**: 配置 CC、CXX、LD_LIBRARY_PATH 环境变量
4. **Conan 配置**: 创建 GCC 15 专用的 Conan profile

### 构建优化
- 使用 `-j$(nproc)` 进行并行编译以加速构建
- 编译完成后清理临时文件以减小镜像大小
- 设置适当的编译标志以处理 GCC 15 的严格检查

### 兼容性考虑
- 保持与现有构建脚本的兼容性
- 设置 libstdc++11 以确保 ABI 兼容性
- 配置标准 C++17 支持

## 使用方法

构建镜像：
```bash
docker build -t milvus-builder-gcc15 build/docker/builder/cpu/ubuntu22.04/
```

## 注意事项

1. GCC 15 编译时间较长，首次构建需要耐心等待
2. 确保有足够的磁盘空间（编译过程需要较多临时空间）
3. 如遇到编译错误，可能需要添加额外的 `#include <cstdint>` 等头文件

## 参考

基于 [alexanderguzhva/milvus_gcc15_conan1](https://github.com/alexanderguzhva/milvus_gcc15_conan1) 的升级方案实现。