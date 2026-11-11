FROM automlorg/autopytorch@sha256:5244f3ee3d923afb0288d7362a1e0c8c2df84628afd6818cd4ceb6df0a87cc82
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-input optuna==3.6.1
WORKDIR /lab
CMD ["sleep", "infinity"]
