
import json
import pickle
import re
import pandas as pd
import streamlit as st
import joblib
import streamlit as st

st.write("DEBUG: App code has started.")
from huggingface_hub import hf_hub_download


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="CropYield | Smart Farming",
    page_icon="🌱",
    layout="wide"
)

# =====================================================
# 2. DESIGN
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #f4f8f3, #edf5f2);
    color: #173b32;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #092c24, #104c38);
}
[data-testid="stSidebar"] * { color: #f1fff5; }

.hero {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(115deg, #103d30, #18794e);
    color: white;
    margin-bottom: 22px;
}
.hero h1 { color: white; font-size: 38px; font-weight: 800; }
.hero p { color: #d9f4df; }

.panel {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #e0ebe2;
    margin-bottom: 16px;
    box-shadow: 0 5px 22px rgba(22,66,48,.05);
}
.panel-title { font-size: 22px; font-weight: 800; color: #103d30; }
.panel-subtitle { font-size: 13px; color: #6a8177; }

.stButton button, .stFormSubmitButton button {
    background: linear-gradient(90deg, #65bd45, #198754);
    color: white !important;
    border: none;
    border-radius: 10px;
    min-height: 44px;
    font-weight: 700;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border: 1px solid #dfebe1;
    border-radius: 14px;
}
.tip {
    background: #e6f5e8;
    border-left: 4px solid #4caa55;
    border-radius: 10px;
    padding: 14px;
    color: #285b3c;
}
.footer {
    text-align: center;
    color: #71877b;
    padding: 24px 0;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


import json
import pickle
import re
import pandas as pd
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="CropYield | Smart Farming",
    page_icon="🌱",
    layout="wide"
)

# =====================================================
# 2. DESIGN
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #f4f8f3, #edf5f2);
    color: #173b32;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #092c24, #104c38);
}
[data-testid="stSidebar"] * { color: #f1fff5; }

.hero {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(115deg, #103d30, #18794e);
    color: white;
    margin-bottom: 22px;
}
.hero h1 { color: white; font-size: 38px; font-weight: 800; }
.hero p { color: #d9f4df; }

.panel {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #e0ebe2;
    margin-bottom: 16px;
    box-shadow: 0 5px 22px rgba(22,66,48,.05);
}
.panel-title { font-size: 22px; font-weight: 800; color: #103d30; }
.panel-subtitle { font-size: 13px; color: #6a8177; }

.stButton button, .stFormSubmitButton button {
    background: linear-gradient(90deg, #65bd45, #198754);
    color: white !important;
    border: none;
    border-radius: 10px;
    min-height: 44px;
    font-weight: 700;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border: 1px solid #dfebe1;
    border-radius: 14px;
}
.tip {
    background: #e6f5e8;
    border-left: 4px solid #4caa55;
    border-radius: 10px;
    padding: 14px;
    color: #285b3c;
}
.footer {
    text-align: center;
    color: #71877b;
    padding: 24px 0;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD MODEL AND OPTIONS
# =====================================================

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Mariaaaah2/crop-yield-model",
        filename="crop_yield_model_compressed.joblib"
    )
    return joblib.load(model_path)

model = load_model()


@st.cache_data
def load_options():
    with open("crop_options.json", "r") as file:
        return json.load(file)


try:
    model = load_model()
    options = load_options()
except Exception as error:
    st.error(
        "Unable to load the model or options file. "
        "Make sure crop_yield_model.pkl and crop_options.json "
        "are in the same folder as app.py."
    )
    st.code(str(error))
    st.stop()


# =====================================================
# 4. SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 14px 0 22px;">
        <div style="font-size: 38px;">🌿</div>
        <div style="font-size: 27px; font-weight: 800;">CropYield</div>
        <div style="font-size: 12px; letter-spacing: 1px;">
            SMART FARMING PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🌾 Yield Predictor", "💬 Farming Assistant"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Simple tools for smarter farming.")


# =====================================================
# 5. HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <h1>Grow smarter. Harvest better. 🌱</h1>
    <p>
        Estimate crop yield and explore basic farming guidance
        in one place.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 6. HELPERS FOR YES/NO VALUES
# =====================================================
def yes_no_label(value):
    """Display friendly labels without changing model input values."""
    normalized = str(value).strip().lower()

    if normalized in ("true", "1", "yes"):
        return "Yes"
    if normalized in ("false", "0", "no"):
        return "No"

    return str(value)


# =====================================================
# 7. YIELD PREDICTOR
# =====================================================
if page == "🌾 Yield Predictor":

    m1, m2, m3 = st.columns(3)
    m1.metric("Prediction tool", "Ready")
    m2.metric("Input features", "9")
    m3.metric("Output unit", "Tons/hectare")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">🌾 Farm details</div>
        <div class="panel-subtitle">
            Enter your conditions to estimate crop yield.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("crop_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            region = st.selectbox("📍 Region", options["Region"])
            soil = st.selectbox("🪴 Soil type", options["Soil_Type"])
            crop = st.selectbox("🌱 Crop", options["Crop"])
            rainfall = st.number_input(
                "🌧️ Rainfall (mm)",
                min_value=0.0,
                value=100.0,
                step=10.0
            )
            temperature = st.number_input(
                "🌡️ Temperature (°C)",
                value=25.0,
                step=1.0
            )

        with col2:
            weather = st.selectbox(
                "⛅ Weather condition",
                options["Weather_Condition"]
            )
            fertilizer = st.selectbox(
                "🧪 Fertilizer used?",
                options["Fertilizer_Used"],
                format_func=yes_no_label
            )
            irrigation = st.selectbox(
                "💧 Irrigation used?",
                options["Irrigation_Used"],
                format_func=yes_no_label
            )
            days = st.number_input(
                "📅 Days to harvest",
                min_value=1,
                value=90,
                step=1
            )

        submitted = st.form_submit_button(
            "🌱 Predict Crop Yield",
            use_container_width=True
        )

    if submitted:
        input_data = pd.DataFrame([{
            "Region": region,
            "Soil_Type": soil,
            "Crop": crop,
            "Rainfall_mm": rainfall,
            "Temperature_Celsius": temperature,
            "Fertilizer_Used": fertilizer,
            "Irrigation_Used": irrigation,
            "Weather_Condition": weather,
            "Days_to_Harvest": days
        }])

        try:
            with st.spinner("Estimating crop yield..."):
                result = float(model.predict(input_data)[0])

            st.session_state["prediction"] = result
            st.session_state["prediction_crop"] = crop

        except Exception as error:
            st.error("Prediction failed. Please check the inputs.")
            st.code(str(error))

    prediction = st.session_state.get("prediction")

    if prediction is not None:
        crop_name = st.session_state.get("prediction_crop", "Selected crop")

        st.markdown("---")
        st.subheader("📊 Your prediction")

        a, b = st.columns(2)
        a.metric("Crop", crop_name)
        b.metric("Estimated yield", f"{prediction:,.2f} tons/hectare")

        st.caption(
            "This is a machine-learning estimate, not a guarantee "
            "of actual harvest."
        )

        report = (
            "CROP YIELD PREDICTION REPORT\n"
            "---------------------------\n"
            f"Crop: {crop_name}\n"
            f"Estimated yield: {prediction:.2f} tons per hectare\n"
            "Note: This is an estimate, not a guaranteed harvest.\n"
        )

        st.download_button(
            "⬇️ Download prediction report",
            data=report,
            file_name="crop_yield_report.txt",
            mime="text/plain"
        )

    st.markdown("""
    <div class="tip">
        <b>💡 Tip:</b> Use realistic rainfall, temperature, and
        farm-management values for a meaningful estimate.
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# 8. FREE RULE-BASED FARMING ASSISTANT
# =====================================================
elif page == "💬 Farming Assistant":

    st.subheader("💬 CropCare — Farming Assistant")
    st.write(
        "Ask a question about common crop problems, soil, watering, "
        "fertilizer, or pests. This basic assistant uses built-in "
        "answers and does not require an API key."
    )

    # Built-in knowledge base. This is not a generative AI model.
    knowledge = [
        {
            "keywords": ["yellow leaf", "yellow leaves", "yellowing", "leaves turn yellow"],
            "answer": (
                "**Yellow leaves can have several causes.** These include "
                "overwatering, poor drainage, nutrient deficiencies, pests, "
                "or natural ageing. Check whether the soil is waterlogged, "
                "inspect the underside of leaves, and compare affected plants "
                "with healthy ones. Yellowing alone cannot identify the exact cause."
            )
        },
        {
            "keywords": ["water", "watering", "irrigation", "how often should i water"],
            "answer": (
                "**Watering depends on the crop, soil, weather, and growth stage.** "
                "Check soil moisture near the roots before watering. Sandy soils "
                "usually dry faster than clay soils. Avoid keeping roots "
                "waterlogged, and follow local crop-specific irrigation guidance."
            )
        },
        {
            "keywords": ["fertilizer", "fertiliser", "manure", "nutrient", "npk"],
            "answer": (
                "**Choose fertilizer based on the crop and soil's nutrient needs.** "
                "A soil test can help identify deficiencies. Follow the product "
                "label and local agricultural recommendations; excess fertilizer "
                "can damage plants and pollute water. Do not apply a guessed dose."
            )
        },
        {
            "keywords": ["pest", "insect", "bugs", "caterpillar", "aphid"],
            "answer": (
                "**Start by identifying the pest and the damage.** Inspect both "
                "sides of leaves and look for eggs, insects, and feeding marks. "
                "Remove heavily affected plant parts when appropriate, protect "
                "beneficial insects, and consult local agricultural experts "
                "before choosing a pesticide."
            )
        },
        {
            "keywords": ["disease", "fungus", "fungal", "spots on leaves", "leaf spot", "mildew", "rot"],
            "answer": (
                "**Leaf spots and rotting can have different causes**, including "
                "fungi, bacteria, pests, or environmental stress. Avoid wetting "
                "foliage unnecessarily, clean tools, and remove infected material "
                "when appropriate. A reliable diagnosis may require a clear photo "
                "and information about the crop and growing conditions."
            )
        },
        {
            "keywords": ["soil", "soil health", "soil test", "soil pH"],
            "answer": (
                "**Healthy soil supports strong roots.** A soil test can help "
                "check pH and nutrient levels. Add organic matter where suitable, "
                "avoid compaction, maintain drainage, and use amendments based "
                "on test results and recommendations for your crop."
            )
        },
        {
            "keywords": ["wilting", "wilt", "drooping"],
            "answer": (
                "**Wilting can be caused by lack of water, damaged roots, "
                "waterlogged soil, heat, or disease.** Check soil moisture before "
                "watering more. If the soil is already wet, adding water may make "
                "the problem worse. Look for root damage or spreading symptoms."
            )
        },
        {
            "keywords": ["rain", "heavy rain", "waterlogging", "flood"],
            "answer": (
                "**After heavy rain, check drainage and standing water.** "
                "Waterlogged roots can lose access to oxygen. Keep drainage "
                "channels clear where safe, avoid walking on saturated soil, "
                "and inspect plants for lodging, root damage, and disease."
            )
        },
        {
            "keywords": ["slow growth", "not growing", "stunted", "small plants"],
            "answer": (
                "**Slow growth may result from poor light, unsuitable temperature, "
                "water stress, root problems, pests, or nutrient deficiencies.** "
                "Check these conditions before adding fertilizer. Compare symptoms "
                "with healthy plants and consider a soil test."
            )
        },
        {
            "keywords": ["rice", "paddy"],
            "answer": (
                "**For rice and paddy,** water management, weed control, suitable "
                "nutrient management, and timely pest monitoring are important. "
                "Practices vary by variety, growth stage, and local conditions. "
                "Follow recommendations from your local agricultural extension service."
            )
        },
        {
            "keywords": ["tomato", "tomatoes"],
            "answer": (
                "**For tomato plants,** maintain consistent soil moisture, ensure "
                "good drainage, and monitor leaves and fruit for disease and pests. "
                "Avoid wetting foliage unnecessarily. Symptoms such as fruit-end "
                "rot or leaf spots need different responses, so check the exact symptoms."
            )
        },
        {
            "keywords": ["hello", "hi", "hey", "good morning"],
            "answer": (
                "Hello, farmer! 🌱 I can share basic guidance about watering, "
                "soil health, fertilizer, yellow leaves, pests, and common crop "
                "problems. Tell me which crop you are growing and what you have noticed."
            )
        },
        {
            "keywords": ["thank you", "thanks"],
            "answer": (
                "You're welcome! 🌱 If you have another farming question, "
                "tell me the crop and describe the problem."
            )
        }
    ]

    def get_farming_answer(question):
        """Match a question to a built-in answer."""
        q = re.sub(r"[^a-z0-9\s]", " ", question.lower())
        q = " ".join(q.split())

        # Match longer phrases first to reduce accidental matches.
        matches = []

        for item in knowledge:
            for keyword in item["keywords"]:
                key = keyword.lower()
                if key in q:
                    matches.append((len(key), item["answer"]))

        if matches:
            matches.sort(key=lambda entry: entry[0], reverse=True)
            return matches[0][1]

        return (
            "I'm a basic farming assistant and don't have a reliable built-in "
            "answer for that question yet. 🌱\n\n"
            "Try asking about yellow leaves, watering, soil health, fertilizer, "
            "pests, wilting, or common rice and tomato care. For a specific "
            "problem, include the crop name, plant age, symptoms, and recent "
            "weather. You can also contact your local agricultural extension "
            "service for crop-specific advice."
        )

    # Keep conversation in the current Streamlit session.
    if "farm_chat" not in st.session_state:
        st.session_state.farm_chat = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 🌱 I'm CropCare, a basic farming assistant. "
                    "What crop are you growing, and how can I help?"
                )
            }
        ]

    st.markdown("#### Try asking")

    suggestions = [
        "Why are my leaves turning yellow?",
        "How should I water my crops?",
        "How can I improve soil health?",
        "What should I check for pests?"
    ]

    suggestion_cols = st.columns(2)

    for index, suggestion in enumerate(suggestions):
        if suggestion_cols[index % 2].button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True
        ):
            st.session_state.farm_chat.append(
                {"role": "user", "content": suggestion}
            )
            st.session_state.farm_chat.append(
                {
                    "role": "assistant",
                    "content": get_farming_answer(suggestion)
                }
            )
            st.rerun()

    st.markdown("---")

    # Display the conversation.
    for message in st.session_state.farm_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input(
        "Ask a farming question..."
    )

    if user_question:
        st.session_state.farm_chat.append(
            {"role": "user", "content": user_question}
        )

        answer = get_farming_answer(user_question)

        st.session_state.farm_chat.append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()

    if st.button("🗑️ Clear chat"):
        st.session_state.farm_chat = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 🌱 I'm CropCare. Ask me about common crop "
                    "problems, watering, fertilizer, pests, or soil health."
                )
            }
        ]
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div class="tip">
        <b>⚠️ Important:</b> This assistant provides general educational
        guidance, not a confirmed diagnosis. For severe crop damage,
        suspected outbreaks, or pesticide decisions, contact a qualified
        local agricultural expert. Always follow product labels and local rules.
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# 9. FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    🌿 CropYield · Crop prediction and basic farming guidance<br>
    Built with Python and Streamlit.
</div>
""", unsafe_allow_html=True)

# =====================================================
# 4. SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 14px 0 22px;">
        <div style="font-size: 38px;">🌿</div>
        <div style="font-size: 27px; font-weight: 800;">CropYield</div>
        <div style="font-size: 12px; letter-spacing: 1px;">
            SMART FARMING PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🌾 Yield Predictor", "💬 Farming Assistant"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Simple tools for smarter farming.")


# =====================================================
# 5. HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <h1>Grow smarter. Harvest better. 🌱</h1>
    <p>
        Estimate crop yield and explore basic farming guidance
        in one place.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 6. HELPERS FOR YES/NO VALUES
# =====================================================
def yes_no_label(value):
    """Display friendly labels without changing model input values."""
    normalized = str(value).strip().lower()

    if normalized in ("true", "1", "yes"):
        return "Yes"
    if normalized in ("false", "0", "no"):
        return "No"

    return str(value)


# =====================================================
# 7. YIELD PREDICTOR
# =====================================================
if page == "🌾 Yield Predictor":

    m1, m2, m3 = st.columns(3)
    m1.metric("Prediction tool", "Ready")
    m2.metric("Input features", "9")
    m3.metric("Output unit", "Tons/hectare")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">🌾 Farm details</div>
        <div class="panel-subtitle">
            Enter your conditions to estimate crop yield.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("crop_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            region = st.selectbox("📍 Region", options["Region"])
            soil = st.selectbox("🪴 Soil type", options["Soil_Type"])
            crop = st.selectbox("🌱 Crop", options["Crop"])
            rainfall = st.number_input(
                "🌧️ Rainfall (mm)",
                min_value=0.0,
                value=100.0,
                step=10.0
            )
            temperature = st.number_input(
                "🌡️ Temperature (°C)",
                value=25.0,
                step=1.0
            )

        with col2:
            weather = st.selectbox(
                "⛅ Weather condition",
                options["Weather_Condition"]
            )
            fertilizer = st.selectbox(
                "🧪 Fertilizer used?",
                options["Fertilizer_Used"],
                format_func=yes_no_label
            )
            irrigation = st.selectbox(
                "💧 Irrigation used?",
                options["Irrigation_Used"],
                format_func=yes_no_label
            )
            days = st.number_input(
                "📅 Days to harvest",
                min_value=1,
                value=90,
                step=1
            )

        submitted = st.form_submit_button(
            "🌱 Predict Crop Yield",
            use_container_width=True
        )

    if submitted:
        input_data = pd.DataFrame([{
            "Region": region,
            "Soil_Type": soil,
            "Crop": crop,
            "Rainfall_mm": rainfall,
            "Temperature_Celsius": temperature,
            "Fertilizer_Used": fertilizer,
            "Irrigation_Used": irrigation,
            "Weather_Condition": weather,
            "Days_to_Harvest": days
        }])

        try:
            with st.spinner("Estimating crop yield..."):
                result = float(model.predict(input_data)[0])

            st.session_state["prediction"] = result
            st.session_state["prediction_crop"] = crop

        except Exception as error:
            st.error("Prediction failed. Please check the inputs.")
            st.code(str(error))

    prediction = st.session_state.get("prediction")

    if prediction is not None:
        crop_name = st.session_state.get("prediction_crop", "Selected crop")

        st.markdown("---")
        st.subheader("📊 Your prediction")

        a, b = st.columns(2)
        a.metric("Crop", crop_name)
        b.metric("Estimated yield", f"{prediction:,.2f} tons/hectare")

        st.caption(
            "This is a machine-learning estimate, not a guarantee "
            "of actual harvest."
        )

        report = (
            "CROP YIELD PREDICTION REPORT\n"
            "---------------------------\n"
            f"Crop: {crop_name}\n"
            f"Estimated yield: {prediction:.2f} tons per hectare\n"
            "Note: This is an estimate, not a guaranteed harvest.\n"
        )

        st.download_button(
            "⬇️ Download prediction report",
            data=report,
            file_name="crop_yield_report.txt",
            mime="text/plain"
        )

    st.markdown("""
    <div class="tip">
        <b>💡 Tip:</b> Use realistic rainfall, temperature, and
        farm-management values for a meaningful estimate.
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# 8. FREE RULE-BASED FARMING ASSISTANT
# =====================================================
elif page == "💬 Farming Assistant":

    st.subheader("💬 CropCare — Farming Assistant")
    st.write(
        "Ask a question about common crop problems, soil, watering, "
        "fertilizer, or pests. This basic assistant uses built-in "
        "answers and does not require an API key."
    )

    # Built-in knowledge base. This is not a generative AI model.
    knowledge = [
        {
            "keywords": ["yellow leaf", "yellow leaves", "yellowing", "leaves turn yellow"],
            "answer": (
                "**Yellow leaves can have several causes.** These include "
                "overwatering, poor drainage, nutrient deficiencies, pests, "
                "or natural ageing. Check whether the soil is waterlogged, "
                "inspect the underside of leaves, and compare affected plants "
                "with healthy ones. Yellowing alone cannot identify the exact cause."
            )
        },
        {
            "keywords": ["water", "watering", "irrigation", "how often should i water"],
            "answer": (
                "**Watering depends on the crop, soil, weather, and growth stage.** "
                "Check soil moisture near the roots before watering. Sandy soils "
                "usually dry faster than clay soils. Avoid keeping roots "
                "waterlogged, and follow local crop-specific irrigation guidance."
            )
        },
        {
            "keywords": ["fertilizer", "fertiliser", "manure", "nutrient", "npk"],
            "answer": (
                "**Choose fertilizer based on the crop and soil's nutrient needs.** "
                "A soil test can help identify deficiencies. Follow the product "
                "label and local agricultural recommendations; excess fertilizer "
                "can damage plants and pollute water. Do not apply a guessed dose."
            )
        },
        {
            "keywords": ["pest", "insect", "bugs", "caterpillar", "aphid"],
            "answer": (
                "**Start by identifying the pest and the damage.** Inspect both "
                "sides of leaves and look for eggs, insects, and feeding marks. "
                "Remove heavily affected plant parts when appropriate, protect "
                "beneficial insects, and consult local agricultural experts "
                "before choosing a pesticide."
            )
        },
        {
            "keywords": ["disease", "fungus", "fungal", "spots on leaves", "leaf spot", "mildew", "rot"],
            "answer": (
                "**Leaf spots and rotting can have different causes**, including "
                "fungi, bacteria, pests, or environmental stress. Avoid wetting "
                "foliage unnecessarily, clean tools, and remove infected material "
                "when appropriate. A reliable diagnosis may require a clear photo "
                "and information about the crop and growing conditions."
            )
        },
        {
            "keywords": ["soil", "soil health", "soil test", "soil pH"],
            "answer": (
                "**Healthy soil supports strong roots.** A soil test can help "
                "check pH and nutrient levels. Add organic matter where suitable, "
                "avoid compaction, maintain drainage, and use amendments based "
                "on test results and recommendations for your crop."
            )
        },
        {
            "keywords": ["wilting", "wilt", "drooping"],
            "answer": (
                "**Wilting can be caused by lack of water, damaged roots, "
                "waterlogged soil, heat, or disease.** Check soil moisture before "
                "watering more. If the soil is already wet, adding water may make "
                "the problem worse. Look for root damage or spreading symptoms."
            )
        },
        {
            "keywords": ["rain", "heavy rain", "waterlogging", "flood"],
            "answer": (
                "**After heavy rain, check drainage and standing water.** "
                "Waterlogged roots can lose access to oxygen. Keep drainage "
                "channels clear where safe, avoid walking on saturated soil, "
                "and inspect plants for lodging, root damage, and disease."
            )
        },
        {
            "keywords": ["slow growth", "not growing", "stunted", "small plants"],
            "answer": (
                "**Slow growth may result from poor light, unsuitable temperature, "
                "water stress, root problems, pests, or nutrient deficiencies.** "
                "Check these conditions before adding fertilizer. Compare symptoms "
                "with healthy plants and consider a soil test."
            )
        },
        {
            "keywords": ["rice", "paddy"],
            "answer": (
                "**For rice and paddy,** water management, weed control, suitable "
                "nutrient management, and timely pest monitoring are important. "
                "Practices vary by variety, growth stage, and local conditions. "
                "Follow recommendations from your local agricultural extension service."
            )
        },
        {
            "keywords": ["tomato", "tomatoes"],
            "answer": (
                "**For tomato plants,** maintain consistent soil moisture, ensure "
                "good drainage, and monitor leaves and fruit for disease and pests. "
                "Avoid wetting foliage unnecessarily. Symptoms such as fruit-end "
                "rot or leaf spots need different responses, so check the exact symptoms."
            )
        },
        {
            "keywords": ["hello", "hi", "hey", "good morning"],
            "answer": (
                "Hello, farmer! 🌱 I can share basic guidance about watering, "
                "soil health, fertilizer, yellow leaves, pests, and common crop "
                "problems. Tell me which crop you are growing and what you have noticed."
            )
        },
        {
            "keywords": ["thank you", "thanks"],
            "answer": (
                "You're welcome! 🌱 If you have another farming question, "
                "tell me the crop and describe the problem."
            )
        }
    ]

    def get_farming_answer(question):
        """Match a question to a built-in answer."""
        q = re.sub(r"[^a-z0-9\s]", " ", question.lower())
        q = " ".join(q.split())

        # Match longer phrases first to reduce accidental matches.
        matches = []

        for item in knowledge:
            for keyword in item["keywords"]:
                key = keyword.lower()
                if key in q:
                    matches.append((len(key), item["answer"]))

        if matches:
            matches.sort(key=lambda entry: entry[0], reverse=True)
            return matches[0][1]

        return (
            "I'm a basic farming assistant and don't have a reliable built-in "
            "answer for that question yet. 🌱\n\n"
            "Try asking about yellow leaves, watering, soil health, fertilizer, "
            "pests, wilting, or common rice and tomato care. For a specific "
            "problem, include the crop name, plant age, symptoms, and recent "
            "weather. You can also contact your local agricultural extension "
            "service for crop-specific advice."
        )

    # Keep conversation in the current Streamlit session.
    if "farm_chat" not in st.session_state:
        st.session_state.farm_chat = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 🌱 I'm CropCare, a basic farming assistant. "
                    "What crop are you growing, and how can I help?"
                )
            }
        ]

    st.markdown("#### Try asking")

    suggestions = [
        "Why are my leaves turning yellow?",
        "How should I water my crops?",
        "How can I improve soil health?",
        "What should I check for pests?"
    ]

    suggestion_cols = st.columns(2)

    for index, suggestion in enumerate(suggestions):
        if suggestion_cols[index % 2].button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True
        ):
            st.session_state.farm_chat.append(
                {"role": "user", "content": suggestion}
            )
            st.session_state.farm_chat.append(
                {
                    "role": "assistant",
                    "content": get_farming_answer(suggestion)
                }
            )
            st.rerun()

    st.markdown("---")

    # Display the conversation.
    for message in st.session_state.farm_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input(
        "Ask a farming question..."
    )

    if user_question:
        st.session_state.farm_chat.append(
            {"role": "user", "content": user_question}
        )

        answer = get_farming_answer(user_question)

        st.session_state.farm_chat.append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()

    if st.button("🗑️ Clear chat"):
        st.session_state.farm_chat = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 🌱 I'm CropCare. Ask me about common crop "
                    "problems, watering, fertilizer, pests, or soil health."
                )
            }
        ]
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div class="tip">
        <b>⚠️ Important:</b> This assistant provides general educational
        guidance, not a confirmed diagnosis. For severe crop damage,
        suspected outbreaks, or pesticide decisions, contact a qualified
        local agricultural expert. Always follow product labels and local rules.
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# 9. FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    🌿 CropYield · Crop prediction and basic farming guidance<br>
    Built with Python and Streamlit.
</div>
""", unsafe_allow_html=True)
