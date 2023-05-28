sphinx-apidoc -feE -o source/ ../src/pymmunomics/
# sphinx-apidoc -fE -o api ../src/pymmunomics
make doctest
sphinx-build -b html source build
