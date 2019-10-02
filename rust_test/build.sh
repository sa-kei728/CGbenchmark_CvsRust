#!/bin/bash
rustc rust.rs #create librust.so
g++ main.cpp -lrust -L. -Xlinker -rpath -Xlinker . #create a.out
