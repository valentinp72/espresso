# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import options, utils
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.lstm import Embedding

from espresso.models.speech_lstm import SpeechLSTMDecoder
from espresso.tasks.speech_recognition import SpeechRecognitionEspressoTask


DEFAULT_MAX_TARGET_POSITIONS = 1e5


@register_model('lstm_lm_espresso')
class LSTMLanguageModelEspresso(FairseqLanguageModel):
    def __init__(self, decoder, args):
        super().__init__(decoder)
        self.is_wordlm = args.is_wordlm

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-freeze-embed', type=lambda x: options.eval_bool(x),
                            help='freeze decoder embeddings')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--adaptive-softmax-cutoff', nargs='*', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--share-embed',
                            type=lambda x: options.eval_bool(x),
                            help='share input and output embeddings')
        parser.add_argument('--is-wordlm', type=lambda x: options.eval_bool(x),
                            help='whether it is word LM or subword LM. Only '
                            'relevant for ASR decoding with LM, and it determines '
                            'how the underlying decoder instance gets the dictionary'
                            'from the task instance when calling cls.build_model()')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if getattr(args, 'max_target_positions', None) is not None:
            max_target_positions = args.max_target_positions
        else:
            max_target_positions = getattr(args, 'tokens_per_sample', DEFAULT_MAX_TARGET_POSITIONS)

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.is_wordlm and hasattr(task, 'word_dictionary'):
            dictionary = task.word_dictionary
        elif isinstance(task, SpeechRecognitionEspressoTask):
            dictionary = task.target_dictionary
        else:
            dictionary = task.source_dictionary

        # separate decoder input embeddings
        pretrained_decoder_embed = None
        if args.decoder_embed_path:
            pretrained_decoder_embed = load_pretrained_embedding_from_file(
                args.decoder_embed_path,
                dictionary,
                args.decoder_embed_dim
            )
        # one last double check of parameter combinations
        if args.share_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                '--share-embed requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        decoder = SpeechLSTMDecoder(
            dictionary=dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attn_type=None,
            encoder_output_units=0,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            max_target_positions=max_target_positions,
        )
        return cls(decoder, args)


@register_model_architecture('lstm_lm_espresso', 'lstm_lm_espresso')
def base_lm_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 48)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_freeze_embed = getattr(args, 'decoder_freeze_embed', False)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 650)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 650)
    args.decoder_rnn_residual = getattr(args, 'decoder_rnn_residual', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    args.share_embed = getattr(args, 'share_embed', False)
    args.is_wordlm = getattr(args, 'is_wordlm', False)


@register_model_architecture('lstm_lm_espresso', 'lstm_lm_wsj')
def lstm_lm_wsj(args):
    base_lm_architecture(args)


@register_model_architecture('lstm_lm_espresso', 'lstm_lm_librispeech')
def lstm_lm_librispeech(args):
    args.dropout = getattr(args, 'dropout', 0.0)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 800)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 800)
    args.decoder_layers = getattr(args, 'decoder_layers', 4)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 800)
    args.share_embed = getattr(args, 'share_embed', True)
    base_lm_architecture(args)


@register_model_architecture('lstm_lm_espresso', 'lstm_lm_swbd')
def lstm_lm_swbd(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1800)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 1800)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 1800)
    args.share_embed = getattr(args, 'share_embed', True)
    base_lm_architecture(args)


@register_model_architecture('lstm_lm_espresso', 'lstm_wordlm_wsj')
def lstm_wordlm_wsj(args):
    args.dropout = getattr(args, 'dropout', 0.35)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1200)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 1200)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 1200)
    args.share_embed = getattr(args, 'share_embed', True)
    args.is_wordlm = True
    base_lm_architecture(args)


#####################
# NEW ARCHITECTURES #
#####################

@register_model_architecture('lstm_lm_espresso', 'lstm_lm_media')
def lstm_lm_librispeech(args):
    args.dropout = getattr(args, 'dropout', 0.0)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 800)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 800)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 800)
    args.share_embed = getattr(args, 'share_embed', True)
    base_lm_architecture(args)
