    # installing R from source on linux
    # https://www.r-bloggers.com/2016/05/installing-r-packages/

    sudo apt-get update
    sudo apt-get install build-essential
    sudo apt-get install libcurl4-openssl-dev libxml2-dev libssl-dev
    sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran
    sudo apt-get install libreadline-dev libbz2-dev liblzma-dev
    sudo apt-get install libpcre3 libpcre3-dev
    sudo apt-get install libjpeg-dev libpng-dev libtiff-dev libx11-dev
    sudo apt-get install libcairo2-dev libtcl8.6 libtk8.6-dev
    sudo apt-get install libglu1-mesa-dev libx11-dev libxmu-dev libxi-dev libglu1-mesa-dev

    cd $HOME
    mkdir R
    cd R
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
    