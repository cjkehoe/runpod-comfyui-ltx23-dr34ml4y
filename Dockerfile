FROM runpod/worker-comfyui:5.7.1-base

WORKDIR /workspace

RUN cp /handler.py /handler_base.py

RUN apt-get update \
  && apt-get install -y --no-install-recommends git \
  && rm -rf /var/lib/apt/lists/*

RUN cd /comfyui && git fetch --tags origin && git checkout v0.16.1
RUN pip install --no-cache-dir --break-system-packages comfy-aimdo

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /workspace/requirements.txt

RUN git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git /comfyui/custom_nodes/ComfyUI-LTXVideo \
  && git -C /comfyui/custom_nodes/ComfyUI-LTXVideo checkout 531512f7286963dc7aff1fd8bf5556e95eae03af \
  && if [ -f /comfyui/custom_nodes/ComfyUI-LTXVideo/requirements.txt ]; then pip install --no-cache-dir --break-system-packages -r /comfyui/custom_nodes/ComfyUI-LTXVideo/requirements.txt; fi

RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /comfyui/custom_nodes/ComfyUI-VideoHelperSuite \
  && git -C /comfyui/custom_nodes/ComfyUI-VideoHelperSuite checkout 2984ec4c4b93292421888f38db74a5e8802a8ff8 \
  && if [ -f /comfyui/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt ]; then pip install --no-cache-dir --break-system-packages -r /comfyui/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt; fi

RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git /comfyui/custom_nodes/ComfyUI-KJNodes \
  && if [ -f /comfyui/custom_nodes/ComfyUI-KJNodes/requirements.txt ]; then pip install --no-cache-dir --break-system-packages -r /comfyui/custom_nodes/ComfyUI-KJNodes/requirements.txt; fi

COPY vendor/ComfyUI-VideoOutputBridge /comfyui/custom_nodes/ComfyUI-VideoOutputBridge

ENV COMFY_ROOT=/comfyui
ENV NETWORK_VOLUME_ROOT=/runpod-volume

COPY asset-manifest.json /workspace/asset-manifest.json
COPY workflow-template.json /workspace/workflow-template.json
COPY workflow_builder.py /workspace/workflow_builder.py
COPY bootstrap.py /workspace/bootstrap.py
COPY handler.py /handler.py

CMD ["/start.sh"]
