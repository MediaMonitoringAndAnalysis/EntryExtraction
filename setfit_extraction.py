from typing import List, Tuple
from nltk.tokenize import sent_tokenize
from setfit import SetFitModel
from itertools import groupby
import gc
from tqdm import tqdm
from punctuators.models import PunctCapSegModelONNX


class EntriesExtractor:
    def __init__(
        self,
        relevancy_model_name: str = "Sfekih/sentence_relevancy_model",
        independance_model_name: str = "Sfekih/sentence_independancy_model",
        max_sentences: int = 5,
        overlap: int = 2,
        punct_model_name: str = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
    ):
        self.punct_extractor = PunctCapSegModelONNX.from_pretrained(punct_model_name)
        self.relevancy_model_name = relevancy_model_name
        self.independance_model_name = independance_model_name
        self.relevancy_model = SetFitModel.from_pretrained(relevancy_model_name)
        self.independance_model = SetFitModel.from_pretrained(independance_model_name)
        self.max_sentences = max_sentences
        self.overlap = overlap
        
    def _flatten_list(self, l: List[List[str]]) -> List[str]:
        """Flatten a list of lists into a single list."""
        return [item for sublist in l for item in sublist]
    
    def _chunk_long_sentences(self, lst: List[str], step) -> List[str]:
        i = 0

        result = []  # Initialize an empty list to store the slices
        while (i + step) <= len(
            lst
        ) + 1:  # Adjust the condition to ensure full coverage of the list
            added_list = lst[i : i + step]
            if len(added_list) > 1:
                result.append(
                    lst[i : i + step]
                )  # Append the slice of the list to the result list
            i += step - self.overlap  # Move i forward by step size minus 1 for the overlap
            
        # flatten the list
        result = [item for sublist in result for item in sublist]

        return result  # Return the list of lists

    def _group_indices_by_value(self, lst: List[int], value=1) -> List[List[int]]:
        groups = []
        for key, group in groupby(enumerate(lst), lambda x: x[1] == value):
            if (
                key
            ):  # Only process groups where the value is equal to the specified value (1)
                indices = [index for index, _ in group]
                
                groups.append(indices)
        return groups
    
    def _clean_entries(self, entries: List[List[str]]) -> List[str]:
        cleaned_entries = []
        for entry in entries:
            if len(entry) > self.max_sentences:
                cleaned_entries.append(self._chunk_long_sentences(entry, self.max_sentences))
            else:
                cleaned_entries.append(entry)
                
        cleaned_entries = [e for e in cleaned_entries if len(str(e)) > 3 and len(e) > 0]

        return cleaned_entries
        

    def _apply_relevancy_models(self, entries: List[str]) -> List[int]:
        relevancy_results = self.relevancy_model.predict(entries)
        return relevancy_results

    def _transform_sublists(self, input_sublists: List[List[int]]) -> List[List[int]]:
        flattened_list = [item for sublist in input_sublists for item in sublist]
        result = []
        current_group = []

        for index, value in enumerate(flattened_list):
            if value == 1:  # Start a new group if we encounter a 1
                if (
                    current_group
                ):  # Add the previous group to the result if it's not empty
                    result.append(current_group)
                current_group = [index]
            else:
                current_group.append(index)  # Add the index to the current group

        if current_group:  # Add the last group if it exists
            result.append(current_group)

        return result

    def _apply_independance_models(self, entries: List[List[str]]) -> List[List[str]]:
        final_sentence_groups = []
        for group in entries:
            independance_results_one_group = self.independance_model.predict(group)
            independance_results_one_group[0] = (
                1  # First sentence is always independent
            )
            final_sentence_groups.append(independance_results_one_group)

        independance_results = self._transform_sublists(final_sentence_groups)
        flat_entries = [item for sublist in entries for item in sublist]

        # Regroup the final entries
        final_regrouped_entries = [
            [flat_entries[i] for i in group] for group in independance_results
        ]

        return final_regrouped_entries

    def _get_list_of_relevant_entries(self, entries: List[str]) -> List[List[str]]:
        """
        Process a single document to extract meaningful entries from its sentences.
        
        This function:
        1. Identifies relevant sentences using the relevancy model
        2. Groups consecutive relevant sentences together
        3. Further splits these groups based on semantic independence
        4. Cleans and formats the final groups
        
        Args:
            entries: List of sentences from a single document
            
        Returns:
            List of lists, where each inner list contains semantically related sentences
        """
        # print(entries)
        relevancy_results = self._apply_relevancy_models(entries)
        gc.collect()
        sentences_indices_groups = self._group_indices_by_value(relevancy_results)
        sentences_groups = [
            [entries[i] for i in group] for group in sentences_indices_groups
        ]
        final_grouped_entries = self._apply_independance_models(sentences_groups)
        final_grouped_entries = self._clean_entries(final_grouped_entries)
        gc.collect()

        return final_grouped_entries
    
    def _redo_punctuation(self, raw_document: str) -> List[str]:
        """Clean the extracted entries from a PDF document."""
        # final_entries: List[str] = []
        document_sentences: List[str] = sent_tokenize(raw_document)
        # print(document_sentences) 

        punc_entries: List[List[str]] = self.punct_extractor.infer(
            document_sentences, apply_sbd=True
        )
        punc_entries = self._flatten_list(punc_entries)
        # print(punc_entries)

        # for entry in punc_entries:
        #     if len(entry) > 5 and entry.count(" ") > 6 and "©" not in entry:
        #         final_entries.append(entry)

        return punc_entries

    def __call__(
        self, documents: List[str]
    ) -> List[List[List[str]]]:
        """
        Main entry point for processing multiple documents.
        
        Takes a list of documents and processes each one to extract meaningful entries.
        For each document:
        1. Splits it into sentences
        2. Processes those sentences to extract and group related information
        
        Args:
            documents: List of document strings to process
            
        Returns:
            List where each element contains the processed entries for one document.
            Each document's entries are organized as lists of related sentence groups.
        """
        final_entries = []
        for document in documents:
            
            entries: List[str] = self._redo_punctuation(document)
            entries: List[List[str]] = self._get_list_of_relevant_entries(entries)
            entries = [" ".join(entry) for entry in entries]
            final_entries.append(entries)
        return final_entries
        