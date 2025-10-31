Re: https://github.com/miso-belica/sumy/issues/204#issuecomment-3461463939 - the
idea is good and it works, but the details of the patch were initially wrong. I have now
corrected the script to be robust, prevent duplicate entries, and successfully integrate
Polish language support into `sumy`.

Here is the final, working `apply_sumy_polish_fix.sh` script:

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
        stemmed_word = self._stemmer(word)
        return stemmed_word if stemmed_word is not None else word


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

# Apply the modification to __init__.py using sed, only if not already present
INIT_FILE="$SUMY_PATH/nlp/stemmers/__init__.py"

# Add import statement for polish stemmer
grep -q "from .polish import stem_word as polish_stemmer" "$INIT_FILE" || sed -i "/from .greek import stem_word as greek_stemmer/a from .polish import stem_word as polish_stemmer" "$INIT_FILE"

# Add polish stemmer to SPECIAL_STEMMERS dictionary
grep -q "        'polish': polish_stemmer," "$INIT_FILE" || sed -i "/        'greek': greek_stemmer,/a \        'polish': polish_stemmer," "$INIT_FILE"

if [ $? -eq 0 ]; then
    echo "Successfully added Polish stemmer to $INIT_FILE (if not already present)"
    echo "Polish language support for sumy has been installed."
else
    echo "Error adding Polish stemmer to $INIT_FILE. There was an issue with sed or grep."
fi
