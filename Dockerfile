# Add a line here to specify the docker image to inherit from.
FROM continuumio/miniconda3

ARG WORK_DIR="/app"

WORKDIR $WORK_DIR

# Add lines here to copy over your src folder and 
# any other files you need in the image (like the saved model).
COPY ./src $WORK_DIR/src
COPY model.h5 $WORK_DIR
COPY README.md $WORK_DIR

# Add a line here to update the base conda environment using the conda.yml. 
COPY conda.yml $WORK_DIR
RUN conda env update -f conda.yml -n base

EXPOSE 5000

RUN echo "Making sure flask is installed..."
RUN python -c "import flask"

# Add a line here to run your app
CMD ["python", "-m", "src.app"]
