"""
Script entry point for board2tex package.
"""

import cv2
import argparse
import board2tex as b2t


def main(board_image_path: str, api_key: str, output_dir: str):

    prompt = """
        Descreva a imagem anexada usando LaTeX e TikZ. Foque na parte que parece ser
        intencional. Ignore traços aleatórios que não fazem sentido. Produza um bloco
        de código em um formato que pode ser facilmente copiado e colado em um documento.
        Forneça também quaisquer dependências necessárias para usá-lo.
    """
    
    parser = b2t.BoardParser()
    transcriber = b2t.GeminiTranscriber(prompt, api_key=api_key)

    board_image = cv2.imread(board_image_path)
    if board_image is None:
        raise ValueError(f"Image at path '{board_image_path}' could not be loaded.")
    
    drawings = parser.parse_board(board_image)
    for idx, drawing in enumerate(drawings):
        latex_code = transcriber.transcribe_to_tex(drawing)
        output_path = f"{output_dir}/drawing_{idx+1}.tex"
        with open(output_path, "w") as f:
            f.write(latex_code)
        print(f"LaTeX code for drawing {idx+1} saved to {output_path}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image of a board and convert it to LaTeX and TikZ code.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("-a", "--api_key", type=str, required=True, help="API key for the generative AI service.")
    parser.add_argument("-o", "--output", type=str, default="output", help="Path to the output LaTeX directory.")
    args = parser.parse_args()

    main(args.image_path, args.api_key, args.output)
