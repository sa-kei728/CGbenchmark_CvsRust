# CGbenchmark_CvsRust
C++ vs Rust by Conjugate Gradient<br>

## ConjugateGradient
[Reference]
<https://proc-cpuinfo.fixstars.com/2019/09/rust-wins-cpp/>
 <https://bitbucket.org/LWisteria/conjugategradient/src/master/><br>

## rust_test
[How to build]<br>
### ① rust code build (create shared library)
```
rustc rust.rs
```
### ② C++ code build
```
g++ main.cpp -lrust -L. -Xlinker -rpath -Xlinker .
```
