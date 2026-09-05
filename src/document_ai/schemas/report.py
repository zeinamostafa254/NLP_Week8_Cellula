from pydantic import BaseModel, Field

class ReportDesignParams(BaseModel):
    """Extracted design parameters from the user's prompt for the PDF report."""
    
    font_family: str = Field(
        default="helvetica", 
        description="The font family to use for the PDF. Use 'helvetica', 'arial', 'times', or 'courier'."
    )
    title_color: tuple[int, int, int] = Field(
        default=(41, 128, 185),
        description="RGB color tuple for the main title, e.g., (255, 0, 0) for red."
    )
    text_color: tuple[int, int, int] = Field(
        default=(50, 50, 50),
        description="RGB color tuple for the main body text."
    )
    heading_color: tuple[int, int, int] = Field(
        default=(41, 128, 185),
        description="RGB color tuple for section headings like 'Citations'."
    )
