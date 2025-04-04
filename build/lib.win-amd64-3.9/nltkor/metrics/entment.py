from nltkor.tag import EspressoTagger

class EntMent :
    
    def __init__(self):
        self.entity_list = []
        self.tagger = EspressoTagger(task = 'ner')
        self.skip = ["*"]
    
    def entity (self,orginal_text,summarized_text) :
    
        self.entity_list = self.tagger.tag(orginal_text)
        self.entity_list = [item for item in self.entity_list if item[1] not in self.skip]
        self.entity_recall(summarized_text)
        
    def entity_recall (self,summarized_text) :
        
        if len(self.entity_list) == 0 :
            return 0.0
        
        summarized_entity_list = self.tagger.tag(summarized_text)
        summarized_entity_list = [item for item in summarized_entity_list  if item[1] not in self.skip]
        recall = len(set(summarized_entity_list)&set(self.entity_list))/len(summarized_entity_list)
        print(recall)
