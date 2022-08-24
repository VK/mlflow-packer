import glob
import os


def setup():
    """
    run pip install for all libs in the model code folder
    """
    cwd = os.getcwd()

    for folder in glob.glob("code/*"):
        # loop all folders
        if os.path.isdir(folder):


            os.chdir(folder)
            # run pip install
            os.system(
                f'pip install .' 
            )  
            os.chdir(cwd)

    os.chdir(cwd)


if __name__ == "__main__":
    setup()
