# -*- coding: utf-8 -*- 

"""
Configuration data for the system.
"""

import os

data_dir = None
FILES = {}

def get_config_paths(directory):
    """Sets the data directory containing the data for the models."""
    assert os.path.isdir(directory), 'Invalid data directory'

    return { key: os.path.join(directory, value) for key, value in [ 
				# cross-task data
				('.', '.'), #for data_dir access
				
				# vocabulary file used as a fallback if a reader doesn't have a specific one
				('vocabulary'                  , 'vocabulary.txt'),
				('type_features'               , 'types-features.npy'),
				('termvectors'                 , 'termvectors.txt'),

				# POS
				('network_pos'                 , 'pos-network.npz'),
				('network_text_pos'            , 'pos-network.txt'),
				('pos_tags'                    , 'pos-tags.txt'),
				('pos_tag_dict'                , 'pos-tags.txt'),
				('pos_co_lexicon'              , 'pos-co-lexicon.txt'),
				('pos_morph_lexicon'           , 'pos-morph-lexicon.txt'),
				('pos_prob_dict'               , 'pos-prob-dict.txt'),
				('suffix'                      , 'suffixes.txt'),
				('suffixes'                    , 'suffixes.txt'),
				('prefix'                      , 'prefixes.txt'),
				('prefixes'                    , 'prefixes.txt'),
				('metadata_pos'                , 'metadata-pos.pickle'),
				('metadata_text_pos'           , 'metadata-pos.txt'),
				('type_features_pos'           , 'types-features-pos.npy'),
				('caps_features_pos'           , 'caps-features-pos.npy'),
				('suffix_features_pos'         , 'suffix-features-pos.npy'),
				('prefix_features_pos'         , 'prefix-features-pos.npy'),
				('vocabulary_pos'              , 'vocabulary-pos.txt'),
				
				# NER
				('network_ner'                 , 'ner-network.npz'),
				('network_text_ner'            , 'ner-network.txt'),
				('ner_tags'                    , 'ner-tags.txt'),
				('ner_tag_dict'                , 'ner-tags.txt'),
				('ner_morph_lexicon'           , 'ner-morph-lexicon.txt'),
				('suffix'                      , 'suffixes.txt'),
				('suffixes'                    , 'suffixes.txt'),
				('prefix'                      , 'prefixes.txt'),
				('prefixes'                    , 'prefixes.txt'),
				('metadata_ner'                , 'metadata-ner.pickle'),
				('metadata_text_ner'           , 'metadata-ner.txt'),
				('type_features_ner'           , 'types-features-ner.npy'),
				('caps_features_ner'           , 'caps-features-ner.npy'),
				('suffix_features_ner'         , 'suffix-features-ner.npy'),
				('prefix_features_ner'         , 'prefix-features-ner.npy'),
				('vocabulary_ner'              , 'vocabulary-ner.txt'),

				# WSD
				('network_wsd'                 , 'wsd-network.npz'),
				('network_text_wsd'            , 'wsd-network.txt'),
				('wsd_tags'                    , 'wsd-tags.txt'),
				('wsd_tag_dict'                , 'wsd-tags.txt'),
				('wsd_morph_lexicon'           , 'wsd-morph-lexicon.txt'),
#				('suffix'                      , 'suffixes.txt'),
#				('suffixes'                    , 'suffixes.txt'),
#				('prefix'                      , 'prefixes.txt'),
#				('prefixes'                    , 'prefixes.txt'),
				('metadata_wsd'                , 'metadata-wsd.pickle'),
				('metadata_text_wsd'           , 'metadata-wsd.txt'),
				('type_features_wsd'           , 'types-features-wsd.npy'),
				('caps_features_wsd'           , 'caps-features-wsd.npy'),
				('suffix_features_wsd'         , 'suffix-features-wsd.npy'),
				('prefix_features_wsd'         , 'prefix-features-wsd.npy'),
				('vocabulary_wsd'              , 'vocabulary-wsd.txt'),

				# dependency
				('network_labeled_dependency', 'ldep-network.npz'),
				('network_text_labeled_dependency', 'ldep-network.txt'),
				('type_features_labeled_dependency', 'types-features-ldep.npy'),
				('caps_features_labeled_dependency', 'caps-features-ldep.npy'),
				('pos_features_labeled_dependency', 'pos-features-ldep.npy'),
				('metadata_labeled_dependency', 'metadata-ldep.pickle'),
				('metadata_text_labeled_dependency', 'metadata-ldep.txt'),
				('dependency_tag_dict', 'dependency-tags.txt'),
				('labeled_dependency_tag_dict', 'dependency-tags.txt'),
				('vocabulary_labeled_dependency', 'vocabulary-ldep.txt'),
				
				('dependency_pos_tags', 'dep-pos-tags.txt'),
				
				('network_unlabeled_dependency', 'udep-network.npz'),
				('network_text_unlabeled_dependency', 'udep-network.txt'),
				('type_features_unlabeled_dependency', 'types-features-udep.npy'),
				('caps_features_unlabeled_dependency', 'caps-features-udep.npy'),
				('pos_features_unlabeled_dependency', 'pos-features-udep.npy'),
				('metadata_unlabeled_dependency', 'metadata-udep.pickle'),
				('metadata_text_unlabeled_dependency', 'metadata-udep.txt'),
				('vocabulary_unlabeled_dependency', 'vocabulary-udep.txt'),
				
				# chunk
				#('chunk_tag_dict'              , 'chunk-tag-dict.pickle'),
				#('chunk_tags'                  , 'chunk-tags.txt'),
				
				# SRL
				('network_srl'                  , 'srl-network.npz'),
				('network_text_srl'             , 'srl-network.txt'),
				#('network_srl_boundary'        , 'srl-id-network.npz'),
				#('network_srl_classify'        , 'srl-class-network.npz'),
				#('network_srl_predicates'      , 'srl-class-predicates.npz'),
				#('srl_iob_tag_dict'            , 'srl-tags.txt'),
				#('srl_iob_tags'                , 'srl-tags.txt'),
				('srl_tags'                    , 'srl-tags.txt'),
				#('srl_classify_tag_dict'       , 'srl-tags.txt'),
				#('srl_classify_tags'           , 'srl-tags.txt'),
				#('srl_predicates_tag_dict'     , 'srl-predicates-tags.txt'),
				#('srl_predicates_tags'         , 'srl-predicates-tags.txt'),
				('type_features_srl'            , 'types-features-srl.npy'),
				('caps_features_srl'            , 'caps-features-srl.npy'),
				('pos_features_srl'             , 'pos-features-srl.npy'),
				#('chunk_features_classify'     , 'chunk-features-class.npy'),
				#('type_features_boundary'      , 'types-features-id.npy'),
				#('caps_features_boundary'      , 'caps-features-id.npy'),
				#('pos_features_boundary'       , 'pos-features-id.npy'),
				#('chunk_features_boundary'     , 'chunk-features-id.npy'),
				#('type_features_classify'      , 'types-features-class.npy'),
				#('caps_features_classify'      , 'caps-features-class.npy'),
				#('pos_features_classify'       , 'pos-features-class.npy'),
				#('chunk_features_classify'     , 'chunk-features-class.npy'),
				#('type_features_1step'         , 'types-features-1step.npy'),
				#('caps_features_1step'         , 'caps-features-1step.npy'),
				#('pos_features_1step'          , 'pos-features-1step.npy'),
				#('chunk_features_1step'        , 'chunk-features-1step.npy'),
				#('type_features_srl_predicates', 'types-features-preds.npy'),
				#('caps_features_srl_predicates', 'caps-features-preds.npy'),
				#('pos_features_srl_predicates' , 'pos-features-preds.npy'),
				('metadata_srl'                , 'metadata-srl.pickle'),
				('metadata_text_srl'            , 'metadata-srl.txt'),
				#('metadata_srl_boundary'       , 'metadata-srl-boundary.pickle'),
				#('metadata_srl_classify'       , 'metadata-srl-classify.pickle'),
				#('metadata_srl_predicates'     , 'metadata-srl-predicates.pickle'),
				('vocabulary_srl', 'vocabulary-srl.txt'),
				#('vocabulary_srl_boundary', 'vocabulary-srl-boundary.txt'),
				#('vocabulary_srl_classify', 'vocabulary-srl-classify.txt'),
				#('vocabulary_srl_predicates', 'vocabulary-srl-predicates.txt')
				]
    }


def set_data_dir(directory):
    """Sets the global data directory containing the data for the models."""
    global data_dir, FILES
    data_dir = directory
    FILES = get_config_paths(directory)

