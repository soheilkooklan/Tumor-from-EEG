"""
Entry point for the graphical interface.

    python main.py

For headless, scriptable runs use the command line instead:

    python -m eegtumor.cli run --manifest cohort.csv --out results/
"""

from eegtumor.gui import main

if __name__ == "__main__":
    main()
