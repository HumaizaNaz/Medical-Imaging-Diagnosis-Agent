Medical Imaging Diagnosis Agent Documentation
medical-imaging-diagnosis-agent09.streamlit.app
Overview
The Medical Imaging Diagnosis Agent is a Streamlit-based web application designed for educational and research purposes to analyze medical images using the Google Gemini 1.5 Flash AI model. It provides a user-friendly interface for uploading medical images (e.g., X-rays, MRIs, CT scans), applying image enhancements, and generating detailed AI-driven analyses tailored to specific medical specialties. The application also integrates with the PubMed API to provide research context and maintains a history of analyses for user reference.
Note: This tool is strictly for educational and research purposes and is not intended for clinical use or medical decision-making.

Key Features

AI-Powered Analysis: Leverages Google Gemini 1.5 Flash for comprehensive medical image analysis.
Multi-Specialty Support: Supports 12+ medical specialties, including Radiology, Cardiology, Neurology, and more.
Image Enhancement: Offers options for contrast, brightness, sharpness, or automatic enhancement to improve image quality for analysis.
Research Integration: Retrieves relevant research papers from PubMed to provide evidence-based context for findings.
Analysis History: Tracks all analyses with metadata, allowing users to review past results.
Export Capabilities: Supports exporting analysis reports as text files or copying to the clipboard.
Real-Time Processing: Delivers instant analysis results with progress indicators.
User-Friendly Interface: Built with Streamlit, featuring a tabbed layout for image analysis, history, statistics, and information.

Supported Imaging Types
The application supports a variety of medical imaging types, including:

X-rays: Chest, bone, dental
CT Scans: All body regions
MRI Images: Brain, spine, joints
Ultrasound: Cardiac, abdominal
Pathology Slides: Histological images
Dermatology: Skin lesions
Ophthalmology: Retinal images

Supported file formats include JPG, JPEG, PNG, BMP, and TIFF, with a maximum file size of 200 MB.

Technical Specifications

AI Model: Google Gemini 1.5 Flash
Supported Formats: JPG, JPEG, PNG, BMP, TIFF
Maximum File Size: 200 MB
Analysis Time: 30–60 seconds average
Research Database: PubMed/NCBI
Enhancement Options: Auto, Contrast, Brightness, Sharpness
Dependencies:
Python 3.8+
streamlit
python-dotenv
pillow
requests
pandas
google-generativeai

Setup Instructions
Prerequisites

Python Environment: Ensure Python 3.8 or higher is installed.
Google Gemini API Key: Obtain an API key from the Google Cloud Console with access to the Gemini 1.5 Flash model.

Installation

Install Dependencies:Run the following command to install required Python packages:
pip install streamlit python-dotenv pillow requests pandas google-generativeai

Set Up Environment Variables:Create a .env file in the project directory with the following content:
GEMINI_API_KEY=your_gemini_api_key_here

Replace your_gemini_api_key_here with your actual Gemini API key.

Save the Application Code:Save the application code in a file named app.py. The code is provided in a separate artifact (artifact_id: 5977f9ef-5a0f-4190-86f9-d3e45c4aeeb6).

Run the Application:Execute the following command to start the Streamlit server:
streamlit run app.py

The application will be accessible in your web browser at <http://localhost:8501>.

Verify API Key:Upon launching, the sidebar will display "✅ GEMINI_API_KEY Connected" if the API key is correctly configured. If you see "❌ GEMINI_API_KEY Missing", check your .env file for errors.

Usage Guidelines
Uploading and Analyzing Images

Upload Image:

Navigate to the "Image Analysis" tab.
Use the file uploader to select a medical image (JPG, JPEG, PNG, BMP, or TIFF).
Ensure the image size is under 200 MB to avoid errors.

Configure Analysis:

Specialty: Select a medical specialty (e.g., Radiology, Cardiology) from the sidebar to focus the analysis.
Image Enhancement: Choose an enhancement type (auto, contrast, brightness, sharpness, or none) to optimize image quality.
Analysis Type: Select Standard Analysis, Quick Assessment, or Detailed Report.
Additional Instructions: Provide specific focus areas (e.g., "Check for fractures") in the text area.
Priority Focus Areas: Select options like Abnormalities or Urgency Assessment to guide the AI.

Run Analysis:

Click the "Analyze Medical Image" button.
A progress bar and status messages will indicate the analysis stages (initializing, processing, saving).
Results will appear in the "Analysis Results" section, including technical assessment, clinical findings, and research context.

Export Results:

Click "Export as Text" to download the analysis as a .txt file.
Click "Copy to Clipboard" to copy the analysis in Markdown format.

Reviewing History

Navigate to the "Analysis History" tab to view past analyses.
Select an analysis from the dropdown to see detailed results.
Export individual analyses using the "Export Selected Analysis" button.

Viewing Statistics

The "Statistics" tab displays:
Specialty Usage Distribution: Bar chart of analyses per specialty.
Analysis Timeline: Line chart of analyses by hour.
Image Format Distribution: Bar chart of image formats used.

Important Notes

Image Quality: Use high-resolution, clear images for optimal results.
File Format: Prefer standard medical imaging formats (e.g., PNG, TIFF).
Clinical Correlation: Always verify AI findings with clinical expertise and patient history.
HIPAA Compliance: Ensure compliance with regulations when handling medical images.

Application Structure
Core Components

Environment Setup:

Loads the GEMINI_API_KEY from a .env file using python-dotenv.
Configures the Google Gemini API with google.generativeai.

Session State:

Maintains analysis_history and current_analysis in Streamlit's session state for tracking analyses.

Helper Functions:

encode_image_to_base64: Converts PIL images to base64 strings for API compatibility.
enhance_image: Applies contrast, brightness, sharpness, or auto enhancements using PIL.ImageEnhance.
get_image_metadata: Extracts image format, mode, size, and transparency.
search_research: Queries the PubMed API for relevant research papers.
get_medical_specialties: Returns a list of supported medical specialties.
get_image_hash: Generates an MD5 hash to track duplicate images.
export_analysis_to_text: Formats analysis results for export.

Image Analysis:

The analyze_medical_image function sends the image and prompt to the Gemini API, processes the response, and simulates function calls (e.g., search_research, assess_urgency) by parsing the output.

Streamlit UI:

Configuration Sidebar: Allows selection of specialty, enhancement type, and analysis settings.
Tabs:
Image Analysis: Upload images, configure analysis, and view results.
Analysis History: Review past analyses with export options.
Statistics: Visualize usage metrics (specialty distribution, timeline, image formats).
Information: Provides details on features, supported imaging types, and disclaimers.

Analysis Output Structure
The AI-generated analysis follows a structured format:

Technical Image Assessment: Image type, quality, anatomical region, and diagnostic adequacy.
Systematic Analysis: Normal and abnormal findings, compared to standard anatomy.
Clinical Findings: Primary, secondary, and incidental findings.
Diagnostic Impression: Likely diagnosis, differential diagnoses, and confidence level.
Clinical Significance: Urgency assessment, follow-up recommendations, and additional imaging needs.
Patient Communication: Simple explanation for patients and next steps.
Research Context: Relevant PubMed papers for identified conditions.

Limitations

AI Accuracy: May miss subtle findings or produce false positives/negatives.
Non-Clinical Use: Not a substitute for professional radiologist interpretation or clinical judgment.
Image Quality Dependency: Analysis quality depends on image resolution and clarity.
Function Calling: Simulated via regex parsing, as Gemini lacks native function-calling support like OpenAI.
API Dependency: Requires a valid Gemini API key and internet connectivity.

Important Medical Disclaimer
This tool is for educational and research purposes only. It should not be used for:

Primary medical diagnosis
Treatment decisions
Emergency medical situations
Replacing professional medical consultation

Always:

Consult qualified healthcare professionals.
Verify AI findings with clinical expertise.
Follow institutional medical protocols.
Consider patient history and clinical context.

Support and Feedback

Technical Issues: Report problems via the application interface or check the console for error messages.
Feature Requests: Submit suggestions for additional specialties, enhancements, or features.
Compliance: Ensure HIPAA compliance when sharing medical images and follow institutional AI usage guidelines.

For further assistance, verify your Gemini API key in the Google Cloud Console and ensure sufficient quota for image analysis tasks.

Future Improvements

Agent Framework Integration: Incorporate a full agent-based approach using Agent, RunConfig, and Runner for enhanced task management and guardrails.
Native Function Calling: If Gemini supports structured function calling in the future, replace regex-based parsing.
Advanced Image Processing: Add support for DICOM files or real-time image preprocessing.
Multi-Model Support: Allow switching between different Gemini models or other AI providers.
Localization: Support non-English prompts and outputs for broader accessibility.

Version Information

Version: 2.0
Last Updated: August 9, 2025
Powered By: Google Gemini 1.5 Flash
License: For educational use only © 2024
"# Medical-Imaging-Diagnosis-Agent" 
