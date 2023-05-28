"""
Questions module
"""

from ..hfpipeline import HFPipeline

class Questions(HFPipeline):
    """
    Runs extractive QA for a series of questions and contexts.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, top_k=1):
        super().__init__("question-answering", path, quantize, gpu, model)
        self.top_k = top_k

    def __call__(self, questions, contexts, workers=0):
        """
        Runs an extractive question-answering model against each question-context pair, finding the best answers.

        Args:
            questions: list of questions
            contexts: list of contexts to pull answers from
            workers: number of concurrent workers to use for processing data, defaults to None

        Returns:
            list of answers
        """

        answers = []

        for x, question in enumerate(questions):
            if question and contexts[x]:
                # Run the QA pipeline
                results = self.pipeline(question=question, context=contexts[x], num_workers=workers, top_k=self.top_k)

                if isinstance(results, list):
                    answer = []
                    for result in results:
                        # Get answer and score
                        single_answer, score = result["answer"], result["score"]

                        # Require score to be at least 0.05
                        if score >= 0.05:
                            answer.append(single_answer)

                    if not answer:
                        answer = None
                else:
                    # Get answer and score
                    answer, score = results["answer"], results["score"]
                    
                    # Require score to be at least 0.05
                    if score < 0.05:
                        answer = None

                # Add result
                answers.append(answer)
            else:
                answers.append(None)

        return answers



