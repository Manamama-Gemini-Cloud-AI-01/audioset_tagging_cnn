# Polish Language Support for Sumy

Version 1.1

This document details the plan how to aid the `sumy` library developer with adding Polish language support to the `sumy` library, including the creation of a custom stemmer using `pystempel` and the integration of Polish stopwords. 

## Problem Statement

The `sumy` library, which relies on `nltk` for stemming and stopwords, does not natively support the Polish language out of the box. The User here, Manamama, used patches and wrote these online. The maintainer of `sumy` asked Manamama to write it (also) as PR instead. 



## Materials:

*   **GitHub Issue:** `https://github.com/miso-belica/sumy/issues/204`
*   **`pystempel` library:** `https://github.com/dzieciou/pystempel/`
*   **Stopwords-iso repository (for Polish stopwords):** `https://github.com/stopwords-iso/stopwords-pl`
*   **Sumy Documentation - How to Add New Language:** `https://miso-belica.github.io/sumy/how-to-add-new-language`

## Solution Approach

To address this, an automated script (`apply_sumy_polish_fix.sh`) was developed to:

1. Locate the `sumy` installation directory.
2. Create a Python module (`polish.py`) for a Polish stemmer using the `pystempel` library.
3. Download a comprehensive list of Polish stopwords.
4. Modify `sumy`'s `__init__.py` file to register the new Polish stemmer.



Status quo: we are testing even this patch for now. But patch is temporary, tactical, PR is  strategic here 

## Artifacts

### 1. `apply_sumy_polish_fix.sh` Script

This script automates the installation process. Its content is as follows:

```bash
#!/bin/bash

# This script applies a patch to the sumy library to add Polish language support.

# Find the sumy installation directory
SUMY_PATH=$(python -c "import sumy; print(sumy.__path__[0])" 2>/dev/null)

if [ -z "$SUMY_PATH" ]; then
    echo "Error: sumy library not found. Please make sure it is installed."
    exit 1
fi

echo "Found sumy at: $SUMY_PATH"

# Create the polish.py stemmer file
STEMMER_FILE="$SUMY_PATH/nlp/stemmers/polish.py"
cat > "$STEMMER_FILE" << EOL
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from pystempel import Stemmer as PystempelStemmer


class PolishStemmer(object):
    """
    Stemmer for Polish language using pystempel.
    """
    def __init__(self):
        self._stemmer = PystempelStemmer.polimorf()

    def stem(self, word):
        return self._stemmer.stem(word)


_stemmer = PolishStemmer()


def stem_word(word):
    return _stemmer.stem(word)

EOL

echo "Created $STEMMER_FILE"

# Create the stopwords directory and download the stopwords file
STOPWORDS_DIR="$SUMY_PATH/data/stopwords"
STOPWORDS_FILE="$STOPWORDS_DIR/polish.txt"
mkdir -p "$STOPWORDS_DIR"
wget -O "$STOPWORDS_FILE" https://raw.githubusercontent.com/stopwords-iso/stopwords-pl/master/stopwords-pl.txt

if [ $? -eq 0 ]; then
    echo "Downloaded Polish stopwords to $STOPWORDS_FILE"
else
    echo "Error downloading Polish stopwords. Please check your internet connection."
    exit 1
fi

# Apply the modification to __init__.py using sed
INIT_FILE="$SUMY_PATH/nlp/stemmers/__init__.py"

sed -i "/        'greek': greek_stemmer,/a \        'polish': polish_stemmer," "$INIT_FILE"

if [ $? -eq 0 ]; then
    echo "Successfully added Polish stemmer to $INIT_FILE"
    echo "Polish language support for sumy has been installed."
else
    echo "Error adding Polish stemmer to $INIT_FILE. The changes may have already been applied or there was an issue with sed."
fi
```

### 2. `polish.py` Stemmer Module

This file is created by the `apply_sumy_polish_fix.sh` script at `$SUMY_PATH/nlp/stemmers/polish.py`. Its content is:

```python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from pystempel import Stemmer as PystempelStemmer


class PolishStemmer(object):
    """
    Stemmer for Polish language using pystempel.
    """
    def __init__(self):
        self._stemmer = PystempelStemmer.polimorf()

    def stem(self, word):
        return self._stemmer.stem(word)


_stemmer = PolishStemmer()


def stem_word(word):
    return _stemmer.stem(word)
```

### 3. Polish Stopwords File

This file is downloaded by the `apply_sumy_polish_fix.sh` script and saved at `$SUMY_PATH/data/stopwords/polish.txt`. The source URI is: `https://raw.githubusercontent.com/stopwords-iso/stopwords-pl/master/stopwords-pl.txt`

### 4. Modified `__init__.py` File

The `__init__.py` file located at `$SUMY_PATH/nlp/stemmers/__init__.py` is modified to include the `polish` stemmer. The relevant changes are:

**Import statement added:**

```python
from .polish import stem_word as polish_stemmer
```

**Entry added to `SPECIAL_STEMMERS` dictionary:**

```python
        'polish': polish_stemmer,
```

## Current Status and Next Steps

Currently, the `apply_sumy_polish_fix.sh` script has been executed, and the Polish stemmer has been added to the `__init__.py` file. However, a duplicate entry for `'polish': polish_stemmer,` was found in the `SPECIAL_STEMMERS` dictionary after the last script execution. This duplicate needs to be removed, and the script should be updated to prevent future duplicates.

**Next steps:**

1. Pip uninstall `sumy`, check if all is clean and `pip install sumy`, for clean slate. 
2. Modify the `apply_sumy_polish_fix.sh` script to ensure the `sed` command only adds the line if it's not already present.
3. Verify the final state of the `__init__.py` file.
4. Update patch online (see the details above)
5. Read the Guidelines (see URL above)
6. Begin preparing for a Pull Request to the `sumy` repository, following their guidelines for adding new languages.