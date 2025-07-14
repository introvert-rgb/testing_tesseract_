# Step 1: Use Ubuntu as base image
FROM ubuntu:22.04

# Step 2: Disable interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Step 3: Install dependencies for Tesseract and Python
# Step 3: Install dependencies for Tesseract and Python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libleptonica-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    zlib1g-dev \
    libicu-dev \
    libpango1.0-dev \
    libcairo2-dev \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    autoconf \
    automake \
    libtool \
    libtool-bin \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*



# Step 4: Clone and build Tesseract from source (5.3.0)
WORKDIR /opt
RUN git clone --branch 5.3.0 https://github.com/tesseract-ocr/tesseract.git && \
    cd tesseract && \
    ./autogen.sh && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Step 5: Add English traineddata model
RUN mkdir -p /usr/local/share/tessdata && \
    wget -O /usr/local/share/tessdata/eng.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata

# Step 6: Copy project files
WORKDIR /app
COPY . /app

# Step 7: Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Step 8: Print Tesseract version for debug
RUN tesseract --version

# Step 9: Expose port for Gradio
EXPOSE 7860

# Step 10: Run the Gradio app
CMD ["python3", "app.py"]
