import logging
import tempfile
from pathlib import Path
from fpdf import FPDF
from langchain_core.prompts import ChatPromptTemplate
from document_ai.llm.model import get_llm
from document_ai.schemas.answer import FinalAnswer
from document_ai.schemas.report import ReportDesignParams

logger = logging.getLogger(__name__)

class ReportAgent:
    """Agent that extracts design parameters from user prompts and generates styled PDFs."""
    
    def __init__(self):
        self._llm = get_llm()
        self._design_extractor = self._llm.with_structured_output(ReportDesignParams)
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a design extraction agent. The user will provide a query. "
             "Your job is to extract any design constraints (colors, fonts) mentioned in the query "
             "to be used for generating a PDF report. If no specific colors are mentioned, use defaults. "
             "If the user says 'red title', extract (255, 0, 0) for title_color, etc."),
            ("human", "{question}")
        ])
        
    def _extract_design(self, question: str) -> ReportDesignParams:
        try:
            logger.info("Extracting design parameters from question...")
            chain = self._prompt | self._design_extractor
            result = chain.invoke({"question": question})
            logger.info(f"Extracted design: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to extract design, using defaults. Error: {e}")
            return ReportDesignParams()

    def generate_pdf(self, final_answer: FinalAnswer, question: str) -> Path:
        """Generates a styled PDF report from a FinalAnswer, styled via the question."""
        
        design = self._extract_design(question)
        
        class PDF(FPDF):
            def header(self):
                self.set_font(design.font_family, "B", 15)
                self.set_text_color(*design.title_color)
                self.cell(0, 10, "Document AI - Research Report", border=False, align="C", new_x="LMARGIN", new_y="NEXT")
                self.ln(5)

            def footer(self):
                self.set_y(-15)
                self.set_font(design.font_family, "I", 8)
                self.set_text_color(128, 128, 128)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")

        pdf = PDF()
        pdf.add_page()
        
        # Add question
        pdf.set_font(design.font_family, "B", 14)
        pdf.set_text_color(*design.heading_color)
        pdf.multi_cell(0, 10, f"Question: {final_answer.question}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
        # Add Answer
        pdf.set_font(design.font_family, "", 12)
        pdf.set_text_color(*design.text_color)
        
        # Clean up some markdown for basic PDF text
        answer_text = final_answer.answer.replace("**", "").replace("*", "")
        pdf.multi_cell(0, 8, answer_text, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)
        
        # Add citations if any
        if final_answer.citations:
            pdf.set_font(design.font_family, "B", 14)
            pdf.set_text_color(*design.heading_color)
            pdf.cell(0, 10, "Citations:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(design.font_family, "", 11)
            pdf.set_text_color(*design.text_color)
            
            for c in final_answer.citations:
                page_str = f", page {c.page}" if c.page else ""
                citation_text = f"[{c.ref_id}] {c.doc}{page_str} (score: {c.score:.3f})"
                pdf.multi_cell(0, 8, citation_text, new_x="LMARGIN", new_y="NEXT")
                pdf.set_font(design.font_family, "I", 10)
                pdf.multi_cell(0, 6, c.snippet.replace("**", ""), new_x="LMARGIN", new_y="NEXT")
                pdf.set_font(design.font_family, "", 11)
                pdf.ln(3)

        # Save to temp file
        temp_file = Path(tempfile.mktemp(suffix=".pdf"))
        pdf.output(temp_file)
        logger.info(f"Generated PDF at {temp_file}")
        
        return temp_file
