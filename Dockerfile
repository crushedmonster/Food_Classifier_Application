# Add a line here to specify the docker image to inherit from.
FROM continuumio/miniconda3

ARG WORK_DIR="/app"

WORKDIR $WORK_DIR

# Create the environment
COPY conda.yml $WORK_DIR 
RUN conda env create -f conda.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "food_classifier", "/bin/bash", "-c"]

# Demonstrate the environment is activated
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

# Add lines here to copy over your src folder and 
# any other files you need in the image (like the saved model).
COPY ./src $WORK_DIR/src
COPY model.h5 $WORK_DIR

EXPOSE 5000

# Add a line here to run your app
ENTRYPOINT ["conda", "run", "-n", "food_classifier", "python", "-m", "src.app"]