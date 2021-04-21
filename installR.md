# installing R from source 
wget http://cran.rstudio.com/src/base/R-4/R-4.0.4.tar.gz
tar xvf R-4.0.4.tar.gz
cd R-4.0.4
./configure --prefix=$HOME/R --enable-R-shlib --with-blas
make && make install


install.packages("devtools")
devtools::install_github("hadley/devtools")
install.packages("laGP")

R CMD javareconf -e


# RPY2
pip3 install .

# VEVN activate
PATH="$VIRTUAL_ENV/bin:$HOME/R/bin:$PATH"

# Make sure g++ gcc gfortran version match if you cannot install laGP