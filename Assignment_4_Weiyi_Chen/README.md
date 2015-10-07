# Running Guide

  - Jump Step 4 if you have python3 installed.
  - Jump to Step 3 if you have python2 installed but without python3 installed.
  - Start from Step 1 if you do not have python installed.

### System Requirements
Make sure your system meets these requirements:
  - Operating system: MacOS 10.7 10.8 10.9 10.10 (it has been tested successfully on these)
  - RAM: 2GB.
  - Disk space: 2GB

### Step 1: Install Command Line Tools 
  - Open terminal, type “xcode-select --install” in terminal (without quotes)
  - A pop-up windows will appear asking you about install tools, choose install tools, wait install to finish
  
### Step 2: Install Homebrew and Python

  ```
  ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
  brew tap samueljohn/python
  brew tap homebrew/science
  brew install python
  ```

### Step 3: Install Python3 and related modules
    
  ```
  pip install python3
  pip3 install numpy
  pip3 install scipy
  pip3 install pandas
  pip3 install lazy
  pip3 install ipython
  pip3 install notebook
  ```

### Step 4: Run the IPython notebook

  ```
  ipython3 notebook
  ```
  Then select the one submitted to view.
     