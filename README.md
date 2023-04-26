# swdatatoolkit
---

### Install in Dev Mode

`setuptools` allows you to install a package without copying any files to your interpreter directory (e.g. the
`site-packages` directory). This allows you to modify your source code and have the changes take effect without you having
to rebuild and reinstall. Here’s how to do it:

    pip install --editable .

This creates a link file in your interpreter site package directory which associate with your source code. For more
information, see ["Development Mode"](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).

### Test

Tests require the `pytest` module to be installed. They can then be excuted with the following command:

    python -m pytest

To allow print statements to be viewed when testing, add the `-s` argument to the end of the `pytest` call.

You can run code coverage report recording if you have installed `coverage`

    coverage run --source=py_src -m pytest

Then print out the report with:

    coverage report -m

### Make Documentation

Making documentation requires you to navigate to the `docs` directory of the project

    cd docs

From there you can make and clean the documents

    make html
    make clean

After that you can navigate to the `docs/_build/html` of the project and serve the html over a 
local web server using:

    python3 -m http.server

Which should serve things over port `8000`

    Serving HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/)


## Acknowledgment

This work was supported in part by NASA Grant Award No. NNH14ZDA001N, NASA/SRAG Direct Contract and two NSF Grant
Awards: No. AC1443061 and AC1931555.

***

This software is distributed using the [GNU General Public License, Version 3](./LICENSE.txt)  
![alt text](./images/gplv3-88x31.png)

***

© 2022 Dustin Kempton, Berkay Aydin, Rafal Angryk

[Data Mining Lab](http://dmlab.cs.gsu.edu/)

[Georgia State University](http://www.gsu.edu/)