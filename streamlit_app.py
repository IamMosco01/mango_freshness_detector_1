"""Streamlit front-end for the mango freshness classifier.

Local:  streamlit run streamlit_app.py
Cloud:  set this file as the entrypoint on share.streamlit.io
"""

import streamlit as st

from mango import CLASS_NAMES, DEFAULT_THRESHOLD, predict

st.set_page_config(page_title='Mango Freshness Detector', page_icon='🥭')


@st.cache_resource
def warm_model():
    """Load the interpreter once per server process, not once per rerun."""
    from mango import get_interpreter
    return get_interpreter()


st.title('🥭 Mango Freshness Detector')
st.caption('Upload a mango photo and the model will call it fresh or rotten.')

with st.sidebar:
    st.header('Settings')
    threshold = st.slider(
        'Rotten threshold', 0.0, 1.0, DEFAULT_THRESHOLD, 0.01,
        help='p(rotten) above this is labelled rotten. The notebook settled on 0.6.',
    )
    st.markdown(
        f'Classes: `{CLASS_NAMES[0]}` = 0, `{CLASS_NAMES[1]}` = 1. '
        'The model outputs one sigmoid value: the probability of **rotten**.'
    )

warm_model()

uploaded = st.file_uploader(
    'Mango image', type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
    label_visibility='collapsed',
)

if uploaded is None:
    st.info('Choose an image to get started.')
    st.stop()

try:
    result = predict(uploaded, threshold=threshold)
except Exception:
    # Pillow's own message embeds the whole file object, which is noise here.
    st.error(f'Could not read **{uploaded.name}** — it may be corrupt or not a real image.')
    st.stop()

left, right = st.columns([1, 1])

with left:
    st.image(uploaded, caption=uploaded.name, width='stretch')

with right:
    if result['label'] == 'fresh':
        st.success('### FRESH')
    else:
        st.error('### ROTTEN')

    st.metric('Confidence', f'{result["confidence"] * 100:.1f}%')
    st.progress(result['rotten_probability'])
    st.caption(
        f'p(rotten) = {result["rotten_probability"]:.4f} '
        f'(threshold {result["threshold"]})'
    )

with st.expander('What the model is doing'):
    st.markdown(
        """
- The image is resized to **224×224** RGB. Pixels stay in 0–255 because the
  `Rescaling` layer is baked into the model.
- A small CNN outputs a single sigmoid value: the probability the mango is rotten.
- Inference runs on the CPU from `mango_classifier.tflite` via LiteRT — no GPU
  and no TensorFlow install needed.
- Move the threshold in the sidebar to trade false positives against false negatives.
        """
    )
