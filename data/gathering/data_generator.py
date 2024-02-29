import random
from const import ConceptType
import typeandactionextraction.bert_tagger.bert_preprocessor as bp

rec_pronouns = ["to"]
act_pronouns = ["by"]


def get_plain_sentence(sen):
    res = ''
    for i in range(len(sen)):
        res += sen[i][0] + ' '
    return res.strip()


class DataGenerator:

    def _fill_sets(self, tagged_instances):
        tagged_obj = bp.data_to_object(tagged_instances)
        self.label_to_tagging = bp.to_list_format(tagged_obj)
        self.originals = bp.to_tuple_format(tagged_obj)
        bos = set()
        actions = set()
        actors = set()
        receivers = set()
        astates = set()
        bostates = set()
        sps = set()
        loc = set()
        for item in self.originals.values():
            last = ('X', 'X')
            for tup in item:
                if last[1] == 'X':
                    last = (tup[0], tup[1])
                elif last[1] == tup[1]:
                    last = (last[0] + ' ' + tup[0], tup[1])
                else:
                    self._switch_case(last, actors, actions, bos, astates, bostates, receivers)
                    last = (tup[0], tup[1])
            self._switch_case(last, actors, actions, bos, astates, bostates, receivers)
        self.bos = list(bos)
        self.actions = list(actions)
        self.actors = list(actors)
        self.receivers = list(receivers)
        self.bostates = list(bostates)
        self.astates = list(astates)
        self.sps = list(sps)
        self.loc = list(loc)

    def get_all_available_entities(self, entity_type):
        if entity_type == ConceptType.ACTOR_NAME.value:
            return self.actors
        if entity_type == ConceptType.ACTION_NAME.value:
            return self.actions
        if entity_type == ConceptType.BO_NAME.value:
            return self.bos
        if entity_type == ConceptType.ACTION_STATUS.value:
            return self.astates
        if entity_type == ConceptType.BO_STATUS.value:
            return self.bostates
        if entity_type == ConceptType.PASSIVE_NAME.value:
            return self.receivers
        if entity_type == ConceptType.SUB_PROCESS.value:
            return self.sps

    def _switch_case(self, item, actors, actions, bos, astates, bostates, receivers):
        """
        helper switch statement
        :return: nothing, adds to the sets
        """
        if item[1] == ConceptType.ACTOR_NAME.value:
            actors.add(item[0])
        if item[1] == ConceptType.ACTION_NAME.value:
            actions.add(item[0])
        if item[1] == ConceptType.BO_NAME.value:
            bos.add(item[0])
        if item[1] == ConceptType.ACTION_STATUS.value:
            astates.add(item[0])
        if item[1] == ConceptType.BO_STATUS.value:
            bostates.add(item[0])
        if item[1] == ConceptType.PASSIVE_NAME.value:
            receivers.add(item[0])

    def __init__(self, tagged_instances):
        self.bos = []
        self.actions = []
        self.actors = []
        self.receivers = []
        self.astates = []
        self.bostates = []
        self.sps = []
        self.loc = []
        self.originals = []
        self.label_to_tagging = {}
        self._fill_sets(tagged_instances)

    def get_defined_sentences(self, num=7):
        sents = [self.originals[sen] for sen in self.label_to_tagging.keys() if 'A' in self.label_to_tagging[sen][1] and 'BO' in self.label_to_tagging[sen][1] and ('ACTOR' in self.label_to_tagging[sen][1] or 'REC' in self.label_to_tagging[sen][1]) and 'BOSTATE' in self.label_to_tagging[sen][1]]
        if num > len(sents):
            num = len(sents)
        return random.sample(sents, num)
