# Add a line here to specify the docker image to inherit from.
FROM continuumio/miniconda3

ARG WORK_DIR="/app"

WORKDIR $WORK_DIR

# Create the environment 
COPY conda.yml $WORK_DIR
RUN conda env create -f conda.yml

# Add lines here to copy over your src folder and 
# any other files you need in the image (like the saved model).
COPY ./src $WORK_DIR/src
COPY model.h5 $WORK_DIR
COPY README.md $WORK_DIR

EXPOSE 8000

# Add a line here to run your app
CMD ["python", "-m", "src.app"]
