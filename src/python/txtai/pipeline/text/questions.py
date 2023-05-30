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
        Runs a extractive question-answering model against each question-context pair, finding the best answers.

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
                answer_list = []
                # This is for handling lists of answers
                if isinstance(results, list):
                    for res in results:
                        # Get answer and score
                        answer, score = res["answer"], res["score"]
                        if score >= 0.05:
                            answer_list.append(answer)
                else:  # This is for handling a single answer
                    answer, score = results["answer"], results["score"]
                    if score >= 0.05:
                        answer_list.append(answer)
                # Add answers
                if answer_list:
                    answers.append(answer_list)
                else:
                    answers.append(None)
            else:
                answers.append(None)

        return answers
