import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image as PILImage, ImageEnhance
import requests
import base64
import io
import google.generativeai as genai
import datetime
import hashlib
from typing import Dict, List
import pandas as pd

# ========== ENVIRONMENT SETUP ==========
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ========== SESSION STATE INITIALIZATION ==========
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

# ========== ENHANCED HELPER FUNCTIONS ==========
def encode_image_to_base64(image):
    """Convert PIL image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def enhance_image(image: PILImage.Image, enhancement_type: str = "auto") -> PILImage.Image:
    """Apply image enhancement techniques for better analysis."""
    if enhancement_type == "contrast":
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.2)
    elif enhancement_type == "brightness":
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.1)
    elif enhancement_type == "sharpness":
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(1.3)
    elif enhancement_type == "auto":
        enhanced = ImageEnhance.Contrast(image).enhance(1.1)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.2)
        return enhanced
    return image

def get_image_metadata(image: PILImage.Image) -> Dict:
    """Extract metadata from the image."""
    return {
        "format": image.format,
        "mode": image.mode,
        "size": image.size,
        "width": image.width,
        "height": image.height,
        "has_transparency": image.mode in ("RGBA", "LA") or "transparency" in image.info
    }

def search_research(topic: str) -> str:
    """Find relevant research papers for the topic using PubMed API."""
    try:
        clean_topic = topic.replace(" ", "+").replace(",", "+")
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={clean_topic}&retmax=5&retmode=json&sort=relevance"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        
        if not ids:
            return "No relevant research found for this topic."
        
        detail_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={','.join(ids[:3])}&retmode=json"
        detail_response = requests.get(detail_url, timeout=10)
        
        if detail_response.ok:
            detail_data = detail_response.json()
            papers = []
            for paper_id in ids[:3]:
                if paper_id in detail_data.get("result", {}):
                    paper_info = detail_data["result"][paper_id]
                    title = paper_info.get("title", "No title available")
                    authors = paper_info.get("authors", [])
                    author_names = [author.get("name", "") for author in authors[:3]]
                    papers.append(f"• {title}\n  Authors: {', '.join(author_names)}\n  Link: https://pubmed.ncbi.nlm.nih.gov/{paper_id}")
            
            return "\n\n".join(papers) if papers else "Research papers found but details unavailable."
        else:
            links = [f"https://pubmed.ncbi.nlm.nih.gov/{id}" for id in ids[:3]]
            return "\n".join(links)
            
    except requests.RequestException as e:
        return f"Error fetching research: {e}"

def get_medical_specialties() -> List[str]:
    """Return list of medical specialties for context."""
    return [
        "Radiology", "Cardiology", "Orthopedics", "Neurology", 
        "Pulmonology", "Gastroenterology", "Oncology", "Emergency Medicine",
        "Dermatology", "Pathology", "Ophthalmology", "Urology"
    ]

def analyze_medical_image(image: PILImage.Image, query: str, specialty: str = "General") -> Dict:
    """Analyze medical image using Gemini API with enhanced prompting."""
    
    metadata = get_image_metadata(image)
    base64_image = encode_image_to_base64(image)
    
    functions = [
        {
            "name": "search_research",
            "description": "Find relevant research papers for medical topics using PubMed API",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The medical topic, condition, or finding to search for research papers"
                    }
                },
                "required": ["topic"]
            }
        },
        {
            "name": "assess_urgency",
            "description": "Assess the urgency level of findings",
            "parameters": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "string",
                        "description": "The medical findings to assess for urgency"
                    }
                },
                "required": ["findings"]
            }
        }
    ]
    
    system_prompt = f"""You are a highly skilled medical imaging expert specializing in {specialty}. 

Analyze the medical image systematically and provide a comprehensive assessment with the following structure:

## 1. Technical Image Assessment
- Image type and quality
- Anatomical region and orientation
- Technical adequacy for diagnosis

## 2. Systematic Analysis
- Normal anatomical structures visible
- Abnormal findings (location, size, characteristics)
- Comparison with normal anatomy

## 3. Clinical Findings
- Primary findings
- Secondary findings
- Incidental findings

## 4. Diagnostic Impression
- Most likely diagnosis
- Differential diagnoses
- Confidence level

## 5. Clinical Significance
- Urgency assessment (use assess_urgency function if concerning findings)
- Recommended follow-up
- Additional imaging if needed

## 6. Patient Communication
- Simple explanation for patient
- What the findings mean
- Next steps in care

## 7. Research Context
- Use search_research function for relevant conditions found
- Provide evidence-based context

Image metadata: {metadata}

Remember: Be thorough but acknowledge limitations. Always recommend correlation with clinical findings and specialist consultation when appropriate."""

    try:
        # Prepare image content for Gemini API
        image_content = {
            "mime_type": "image/png",
            "data": base64.b64decode(base64_image)
        }
        
        # Initial prompt with image and text
        content = [
            {"text": system_prompt},
            {"text": f"Please analyze this {specialty.lower()} medical image: {query}"},
            {"inline_data": image_content}
        ]
        
        # First API call to analyze the image
        response = model.generate_content(
            content,
            generation_config={
                "max_output_tokens": 2000,
                "temperature": 0.1
            }
        )
        
        # Handle function calls (Gemini API doesn't natively support function calling like OpenAI,
        # so we simulate it by parsing the response and calling functions if requested)
        function_results = []
        analysis_text = response.text
        
        # Check for function call patterns in the response (e.g., "call search_research('topic')")
        import re
        function_calls = re.findall(r"call (\w+)\('([^']+)'\)", analysis_text)
        
        for func_name, func_arg in function_calls:
            if func_name == "search_research":
                research_results = search_research(func_arg)
                function_results.append(f"Research for '{func_arg}':\n{research_results}")
                analysis_text += f"\n\n{function_results[-1]}"
            elif func_name == "assess_urgency":
                urgency_assessment = f"Urgency assessment for: {func_arg}\nThis requires clinical correlation and appropriate follow-up based on institutional protocols."
                function_results.append(urgency_assessment)
                analysis_text += f"\n\n{function_results[-1]}"
        
        return {
            "analysis": analysis_text,
            "metadata": metadata,
            "function_results": function_results,
            "timestamp": datetime.datetime.now(),
            "specialty": specialty
        }
        
    except Exception as e:
        return {
            "analysis": f"Error analyzing image: {str(e)}",
            "metadata": metadata,
            "function_results": [],
            "timestamp": datetime.datetime.now(),
            "specialty": specialty,
            "error": True
        }

def save_analysis_to_history(analysis_result: Dict, image_hash: str):
    """Save analysis to session history."""
    history_entry = {
        "id": len(st.session_state.analysis_history) + 1,
        "timestamp": analysis_result["timestamp"],
        "image_hash": image_hash,
        "specialty": analysis_result["specialty"],
        "analysis": analysis_result["analysis"],
        "metadata": analysis_result["metadata"],
        "has_error": analysis_result.get("error", False)
    }
    st.session_state.analysis_history.append(history_entry)

def get_image_hash(image: PILImage.Image) -> str:
    """Generate hash for image to track duplicates."""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    return hashlib.md5(img_bytes.getvalue()).hexdigest()

def export_analysis_to_text(analysis_data: Dict) -> str:
    """Export analysis to formatted text."""
    export_text = f"""
MEDICAL IMAGING ANALYSIS REPORT
===============================

Timestamp: {analysis_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
Specialty: {analysis_data['specialty']}
Image Size: {analysis_data['metadata']['width']}x{analysis_data['metadata']['height']}
Image Format: {analysis_data['metadata']['format']}

ANALYSIS:
{analysis_data['analysis']}

---
Generated by Medical Imaging Diagnosis Agent
For Educational Purposes Only - Not for Clinical Use
"""
    return export_text

# ========== STREAMLIT UI CONFIGURATION ==========
st.set_page_config(
    page_title="Medical Imaging Diagnosis Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .analysis-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ========== ENHANCED SIDEBAR ==========
with st.sidebar:
    st.markdown("### 🛠️ Configuration")
    
    if GEMINI_API_KEY:
        st.success("✅ GEMINI_API_KEY Connected")
    else:
        st.error("❌ GEMINI_API_KEY Missing")
    
    st.divider()
    
    st.markdown("### 🎯 Medical Specialty")
    specialty = st.selectbox(
        "Select specialty focus:",
        ["General"] + get_medical_specialties(),
        help="Choose a medical specialty for focused analysis"
    )
    
    st.markdown("### 🖼️ Image Processing")
    enhancement = st.selectbox(
        "Enhancement type:",
        ["auto", "none", "contrast", "brightness", "sharpness"],
        help="Apply image enhancement for better analysis"
    )
    
    st.markdown("### ⚙️ Analysis Settings")
    include_research = st.checkbox("Include research papers", value=True)
    detailed_analysis = st.checkbox("Detailed technical analysis", value=True)
    
    st.divider()
    
    st.markdown("### 📊 Session Summary")
    if st.session_state.analysis_history:
        total_analyses = len(st.session_state.analysis_history)
        successful_analyses = sum(1 for a in st.session_state.analysis_history if not a.get("has_error", False))
        
        st.metric("Total Analyses", total_analyses)
        st.metric("Successful", successful_analyses)
        st.metric("Success Rate", f"{(successful_analyses/total_analyses)*100:.1f}%")
        
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.session_state.current_analysis = None
            st.success("History cleared!")
            st.rerun()
    else:
        st.info("No analyses performed yet")
    
    st.divider()
    
    st.markdown("### 🚀 Quick Actions")
    if st.button("🔄 Refresh App"):
        st.rerun()

# ========== MAIN HEADER ==========
st.markdown("""
<div class="main-header">
    <h1>🏥 Medical Imaging Diagnosis Agent</h1>
    <p>Professional AI-powered medical image analysis using Google Gemini</p>
</div>
""", unsafe_allow_html=True)

# ========== MAIN INTERFACE WITH TABS ==========
tab1, tab2, tab3, tab4 = st.tabs(["📤 Image Analysis", "📊 Analysis History", "📈 Statistics", "ℹ️ Information"])

# ========== TAB 1: IMAGE ANALYSIS ==========
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Medical Image")
        uploaded_file = st.file_uploader(
            "Choose a medical image file",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Supported formats: JPG, JPEG, PNG, BMP, TIFF (Max size: 200MB)"
        )
        
        if uploaded_file:
            try:
                file_size = len(uploaded_file.getvalue())
                if file_size > 200 * 1024 * 1024:
                    st.error("File size too large. Please upload an image smaller than 200MB.")
                else:
                    image = PILImage.open(uploaded_file)
                    st.image(image, caption="📸 Original Image", use_container_width=True)
                    
                    metadata = get_image_metadata(image)
                    with st.expander("📋 Image Information"):
                        col_meta1, col_meta2 = st.columns(2)
                        with col_meta1:
                            st.write(f"**Format:** {metadata['format']}")
                            st.write(f"**Mode:** {metadata['mode']}")
                            st.write(f"**Size:** {metadata['width']} × {metadata['height']}")
                        with col_meta2:
                            st.write(f"**File Size:** {file_size / (1024*1024):.2f} MB")
                            st.write(f"**Aspect Ratio:** {metadata['width']/metadata['height']:.2f}")
                            st.write(f"**Transparency:** {'Yes' if metadata['has_transparency'] else 'No'}")
                    
                    if enhancement != "none":
                        with st.spinner("Enhancing image..."):
                            enhanced_image = enhance_image(image, enhancement)
                            if enhancement != "auto":
                                st.image(enhanced_image, caption=f"🔧 Enhanced ({enhancement})", use_container_width=True)
                            image = enhanced_image
                    
            except Exception as e:
                st.error(f"❌ Error loading image: {e}")
                image = None
    
    with col2:
        if uploaded_file and 'image' in locals() and image is not None:
            st.markdown("### ⚙️ Analysis Configuration")
            
            analysis_type = st.radio(
                "Analysis Type:",
                ["Standard Analysis", "Quick Assessment", "Detailed Report"],
                help="Choose the depth of analysis required"
            )
            
            custom_query = st.text_area(
                "Additional Instructions (Optional):",
                placeholder="e.g., Focus on cardiac structures, look for signs of pneumonia, check for fractures...",
                height=100,
                help="Provide specific instructions for the AI analysis"
            )
            
            priority_focus = st.multiselect(
                "Priority Focus Areas:",
                ["Abnormalities", "Measurements", "Comparison with Normal", "Urgency Assessment", "Treatment Recommendations"],
                default=["Abnormalities", "Urgency Assessment"]
            )
            
            st.markdown("---")
            if st.button("🔍 **Analyze Medical Image**", type="primary", use_container_width=True):
                base_query = "Please provide a comprehensive medical analysis of this image."
                if custom_query:
                    base_query += f" Additional focus: {custom_query}"
                
                if priority_focus:
                    base_query += f" Pay special attention to: {', '.join(priority_focus)}"
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔄 Initializing analysis...")
                    progress_bar.progress(20)
                    
                    status_text.text("🧠 Processing with AI...")
                    progress_bar.progress(50)
                    
                    result = analyze_medical_image(image, base_query, specialty)
                    progress_bar.progress(80)
                    
                    status_text.text("💾 Saving results...")
                    st.session_state.current_analysis = result
                    
                    image_hash = get_image_hash(image)
                    save_analysis_to_history(result, image_hash)
                    progress_bar.progress(100)
                    
                    status_text.text("✅ Analysis complete!")
                    import time
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    progress_bar.empty()
                    status_text.empty()
    
    if st.session_state.current_analysis:
        st.markdown("---")
        st.markdown("### 🧾 Analysis Results")
        
        result = st.session_state.current_analysis
        
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        with col_info1:
            st.metric("Specialty", result["specialty"])
        with col_info2:
            st.metric("Timestamp", result["timestamp"].strftime("%H:%M:%S"))
        with col_info3:
            st.metric("Image Size", f"{result['metadata']['width']}×{result['metadata']['height']}")
        with col_info4:
            status = "❌ Error" if result.get("error", False) else "✅ Success"
            st.metric("Status", status)
        
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown(result["analysis"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            if st.button("📄 Export as Text"):
                export_text = export_analysis_to_text(result)
                st.download_button(
                    label="💾 Download Report",
                    data=export_text,
                    file_name=f"medical_analysis_{result['timestamp'].strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col_export2:
            if st.button("📋 Copy to Clipboard"):
                st.code(result["analysis"], language="markdown")

# ========== TAB 2: ANALYSIS HISTORY ==========
with tab2:
    st.markdown("### 📊 Analysis History")
    
    if st.session_state.analysis_history:
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        total_analyses = len(st.session_state.analysis_history)
        successful_analyses = sum(1 for a in st.session_state.analysis_history if not a.get("has_error", False))
        specialties_used = len(set(a["specialty"] for a in st.session_state.analysis_history))
        
        with col_stat1:
            st.metric("Total Analyses", total_analyses)
        with col_stat2:
            st.metric("Successful", successful_analyses)
        with col_stat3:
            st.metric("Success Rate", f"{(successful_analyses/total_analyses)*100:.1f}%")
        with col_stat4:
            st.metric("Specialties Used", specialties_used)
        
        st.markdown("---")
        
        history_data = []
        for entry in st.session_state.analysis_history:
            history_data.append({
                "ID": entry["id"],
                "Timestamp": entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "Specialty": entry["specialty"],
                "Status": "❌ Error" if entry["has_error"] else "✅ Success",
                "Image Size": f"{entry['metadata']['width']}×{entry['metadata']['height']}",
                "Format": entry['metadata']['format']
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("### 🔍 Detailed Analysis View")
        analysis_options = [f"Analysis {entry['id']} - {entry['timestamp'].strftime('%H:%M:%S')} ({entry['specialty']})" 
                          for entry in st.session_state.analysis_history]
        
        selected_analysis = st.selectbox("Select analysis to view:", analysis_options)
        
        if selected_analysis:
            entry_id = int(selected_analysis.split()[1])
            selected_entry = next(entry for entry in st.session_state.analysis_history if entry["id"] == entry_id)
            
            st.markdown("#### Analysis Details")
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.markdown(selected_entry["analysis"])
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("📄 Export Selected Analysis"):
                export_text = export_analysis_to_text(selected_entry)
                st.download_button(
                    label="💾 Download Selected Report",
                    data=export_text,
                    file_name=f"medical_analysis_{selected_entry['id']}_{selected_entry['timestamp'].strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    else:
        st.info("📝 No analysis history available. Upload and analyze an image to get started.")

# ========== TAB 3: STATISTICS ==========
with tab3:
    st.markdown("### 📈 Usage Statistics")
    
    if st.session_state.analysis_history:
        specialty_counts = {}
        for entry in st.session_state.analysis_history:
            specialty = entry["specialty"]
            specialty_counts[specialty] = specialty_counts.get(specialty, 0) + 1
        
        if specialty_counts:
            st.markdown("#### Specialty Usage Distribution")
            specialty_df = pd.DataFrame(list(specialty_counts.items()), columns=["Specialty", "Count"])
            st.bar_chart(specialty_df.set_index("Specialty"))
        
        st.markdown("#### Analysis Timeline")
        timeline_data = []
        for entry in st.session_state.analysis_history:
            timeline_data.append({
                "Date": entry["timestamp"].strftime("%Y-%m-%d"),
                "Hour": entry["timestamp"].hour,
                "Success": 1 if not entry.get("has_error", False) else 0
            })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            hourly_usage = timeline_df.groupby("Hour").size()
            st.line_chart(hourly_usage)
        
        st.markdown("#### Image Format Distribution")
        format_counts = {}
        for entry in st.session_state.analysis_history:
            img_format = entry["metadata"]["format"]
            format_counts[img_format] = format_counts.get(img_format, 0) + 1
        
        if format_counts:
            format_df = pd.DataFrame(list(format_counts.items()), columns=["Format", "Count"])
            st.bar_chart(format_df.set_index("Format"))
    else:
        st.info("📊 No statistics available yet. Perform some analyses to see usage statistics.")

# ========== TAB 4: INFORMATION ==========
with tab4:
    st.markdown("### ℹ️ About Medical Imaging Diagnosis Agent")
    
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        st.markdown("""
        #### 🚀 Key Features
        - **AI-Powered Analysis**: Google Gemini model
        - **Multi-Specialty Support**: 12+ medical specialties
        - **Image Enhancement**: Automatic optimization
        - **Research Integration**: PubMed paper lookup
        - **Analysis History**: Track all analyses
        - **Export Capabilities**: Download reports
        - **Real-time Processing**: Instant results
        """)
    
    with col_feat2:
        st.markdown("""
        #### 🏥 Supported Imaging Types
        - **X-rays**: Chest, bone, dental
        - **CT Scans**: All body regions
        - **MRI Images**: Brain, spine, joints
        - **Ultrasound**: Cardiac, abdominal
        - **Pathology Slides**: Histological images
        - **Dermatology**: Skin lesions
        - **Ophthalmology**: Retinal images
        """)
    
    st.markdown("#### 🔧 Technical Specifications")
    tech_specs = {
        "AI Model": "Google Gemini 1.5 Flash",
        "Supported Formats": "JPG, JPEG, PNG, BMP, TIFF",
        "Maximum File Size": "200 MB",
        "Analysis Time": "30-60 seconds average",
        "Research Database": "PubMed/NCBI",
        "Enhancement Options": "Auto, Contrast, Brightness, Sharpness"
    }
    
    for spec, value in tech_specs.items():
        st.write(f"**{spec}:** {value}")
    
    st.markdown("#### 📋 Usage Guidelines")
    st.markdown("""
    1. **Image Quality**: Upload high-resolution, clear images for best results
    2. **File Format**: Use standard medical imaging formats when possible
    3. **Specialty Selection**: Choose the most relevant medical specialty
    4. **Custom Instructions**: Provide specific areas of focus for targeted analysis
    5. **Multiple Views**: Analyze different views/angles for comprehensive assessment
    6. **Clinical Correlation**: Always correlate AI findings with clinical presentation
    """)
    
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown("""
    #### ⚠️ Important Medical Disclaimer
    
    **This tool is designed for educational and research purposes only.**
    
    **DO NOT USE for:**
    - Primary medical diagnosis
    - Treatment decisions
    - Emergency medical situations
    - Replacing professional medical consultation
    
    **ALWAYS:**
    - Consult qualified healthcare professionals
    - Verify AI findings with clinical expertise
    - Follow institutional medical protocols
    - Consider patient history and clinical context
    
    **Limitations:**
    - AI analysis may miss subtle findings
    - Cannot replace radiologist interpretation
    - May produce false positives/negatives
    - Limited by image quality and type
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("#### 📞 Support & Feedback")
    st.markdown("""
    For technical support, feature requests, or feedback:
    - Report issues through the application interface
    - Ensure HIPAA compliance when sharing medical images
    - Follow institutional guidelines for AI tool usage
    """)

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em; padding: 1rem;'>
    🏥 Medical Imaging Diagnosis Agent v2.0 | 
    Powered by Google Gemini | 
    For Educational Use Only | 
    © 2024
</div>
""", unsafe_allow_html=True)