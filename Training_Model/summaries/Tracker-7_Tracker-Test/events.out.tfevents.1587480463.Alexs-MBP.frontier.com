       �K"	  �����Abrain.Event:2yi��F      n՚R	������A*�
�
HyperparametersB�BtrainerBppoB
batch_sizeB1024BbetaB0.005Bbuffer_sizeB10240BepsilonB0.2Bhidden_unitsB128BlambdB0.95Blearning_rateB0.0003Blearning_rate_scheduleBlinearB	max_stepsB5.0e5Bmemory_sizeB128B	normalizeBFalseB	num_epochB3B
num_layersB2Btime_horizonB64Bsequence_lengthB64Bsummary_freqB10000Buse_recurrentBFalseBvis_encode_typeBsimpleBreward_signalsB/{'extrinsic': {'strength': 1.0, 'gamma': 0.99}}Bsummary_pathBTracker-7_Tracker-TestB
model_pathB./models/Tracker-7/Tracker-TestBkeep_checkpointsB5J

text>���%       �6�	�`�-§�A�N*

Policy/Entropyş�?�1�6       OW��	Ui�-§�A�N*(
&
Policy/Extrinsic Value Estimate�����	l�4       ^3\	�l�-§�A�N*&
$
Environment/Cumulative RewardUA�ћ�1       ����	�n�-§�A�N*#
!
Environment/Episode Length  �Bg��.       ��W�	3q�-§�A�N* 

Policy/Extrinsic Reward��XA�R�a7       ���Y	�ç�A��*(
&
Policy/Extrinsic Value Estimate/NO>�t/       m]P	R�ç�A��* 

Policy/Extrinsic Reward22rA��!&       sO� 	��ç�A��*

Policy/Entropy{�?�ED)       7�_ 	��ç�A��*

Losses/Value LossQ��@��*       ����	r�ç�A��*

Losses/Policy Lossr`�<���,       ���E	/�ç�A��*

Policy/Learning Rate���9�ˡ95       ��]�	��ç�A��*&
$
Environment/Cumulative Reward��wA����2       $V�	U�ç�A��*#
!
Environment/Episode Length  �B��D7       ���Y	��ħ�A��*(
&
Policy/Extrinsic Value EstimateM��>5�Q&       sO� 	:�ħ�A��*

Policy/EntropyR;�?y�n5       ��]�	�A�ħ�A��*&
$
Environment/Cumulative Reward��YA��T�2       $V�	Ym�ħ�A��*#
!
Environment/Episode Length  �B}��/       m]P	�r�ħ�A��* 

Policy/Extrinsic Reward��YA����)       7�_ 	Kv�ħ�A��*

Losses/Value LossA�@_f��*       ����	�}�ħ�A��*

Losses/Policy LossQ��<%/Y,       ���E	s��ħ�A��*

Policy/Learning Rate-��9|�J�7       ���Y	x��ŧ�A��*(
&
Policy/Extrinsic Value Estimatea?i#M'/       m]P	'g�ŧ�A��* 

Policy/Extrinsic Reward�QAA͒�F&       sO� 	�j�ŧ�A��*

Policy/Entropy�p�?Q_m5       ��]�	Lm�ŧ�A��*&
$
Environment/Cumulative Reward��?A��a2       $V�	$�ŧ�A��*#
!
Environment/Episode Length  �B�N)       7�_ 	��ŧ�A��*

Losses/Value Loss�	�@���*       ����	,��ŧ�A��*

Losses/Policy Loss���<���9,       ���E	��ŧ�A��*

Policy/Learning Rate�Y�9��7       ���Y	����ŧ�AІ*(
&
Policy/Extrinsic Value Estimate��B?�V��/       m]P	���ŧ�AІ* 

Policy/Extrinsic Reward##�Am�
e&       sO� 	���ŧ�AІ*

Policy/EntropyU~�?XlDM5       ��]�	���ŧ�AІ*&
$
Environment/Cumulative Reward ��A��-2       $V�	u��ŧ�AІ*#
!
Environment/Episode Length  �BS��)       7�_ 	���ŧ�AІ*

Losses/Value Loss���@����*       ����	"��ŧ�AІ*

Losses/Policy Loss���<�A�e,       ���E	�#��ŧ�AІ*

Policy/Learning Rate
�9�5q7       ���Y	�]�Ƨ�A��*(
&
Policy/Extrinsic Value Estimate�8�?��G)&       sO� 	�A]�Ƨ�A��*

Policy/Entropyʈ�?����5       ��]�	�D]�Ƨ�A��*&
$
Environment/Cumulative RewardmA>��,2       $V�	G]�Ƨ�A��*#
!
Environment/Episode Length  �B�H��/       m]P	�H]�Ƨ�A��* 

Policy/Extrinsic Reward��mA��{�)       7�_ 	�J]�Ƨ�A��*

Losses/Value Loss�>A	��*       ����	ML]�Ƨ�A��*

Losses/Policy Loss���<0N�7,       ���E	#O]�Ƨ�A��*

Policy/Learning Rate���9)3&7       ���Y	O��ǧ�A�*(
&
Policy/Extrinsic Value Estimate��?I�/       m]P	)#�ǧ�A�* 

Policy/Extrinsic RewardW?XAx�Wr&       sO� 	�%�ǧ�A�*

Policy/Entropy���?�,�55       ��]�	�'�ǧ�A�*&
$
Environment/Cumulative RewardZZZA��K2       $V�	�4�ǧ�A�*#
!
Environment/Episode Length  �BLF��)       7�_ 	H8�ǧ�A�*

Losses/Value LossZd�@/r�*       ����	7:�ǧ�A�*

Losses/Policy Loss���<��-,       ���E	�;�ǧ�A�*

Policy/Learning Rate�w�9@�˕7       ���Y	~r˧ȧ�A��*(
&
Policy/Extrinsic Value Estimate���?0�/       m]P	L�˧ȧ�A��* 

Policy/Extrinsic Reward��JA���&       sO� 	��˧ȧ�A��*

Policy/Entropy`��?S1��5       ��]�	ӣ˧ȧ�A��*&
$
Environment/Cumulative Reward��HA7{A�2       $V�	ץ˧ȧ�A��*#
!
Environment/Episode Length  �Bk�	�)       7�_ 	��˧ȧ�A��*

Losses/Value Loss"0�@�4Z}*       ����	L�˧ȧ�A��*

Losses/Policy Loss��<��[,       ���E	�˧ȧ�A��*

Policy/Learning Rate�#�9���7       ���Y	���ɧ�A��*(
&
Policy/Extrinsic Value Estimate[�?>�W&       sO� 	��ɧ�A��*

Policy/Entropy�c�?~l�5       ��]�	{�ɧ�A��*&
$
Environment/Cumulative RewardqA%*2       $V�	��ɧ�A��*#
!
Environment/Episode Length  �BKET/       m]P	M/�ɧ�A��* 

Policy/Extrinsic RewardT�pA���)       7�_ 	i5�ɧ�A��*

Losses/Value Loss�g�@%�W*       ����	BC�ɧ�A��*

Losses/Policy Loss$i�<��p,       ���E	8G�ɧ�A��*

Policy/Learning Rate�܂9葙�7       ���Y	~:�wʧ�A��*(
&
Policy/Extrinsic Value Estimate�6�?�g�/       m]P	Y�wʧ�A��* 

Policy/Extrinsic RewardR�?A0��&       sO� 	�g�wʧ�A��*

Policy/Entropy6Z�?���5       ��]�	&l�wʧ�A��*&
$
Environment/Cumulative Reward��:A(�;k2       $V�	uo�wʧ�A��*#
!
Environment/Episode Length  �B^�H)       7�_ 	qr�wʧ�A��*

Losses/Value Loss>h�@\5�}*       ����	u�wʧ�A��*

Losses/Policy Loss:��<���,       ���E	�w�wʧ�A��*

Policy/Learning Rate�9y�7       ���Y	��7_˧�A��*(
&
Policy/Extrinsic Value EstimateKg�?iJ��/       m]P	�8_˧�A��* 

Policy/Extrinsic Reward<<lA�I�&       sO� 	�8_˧�A��*

Policy/Entropyȁ�?-@�V5       ��]�	S8_˧�A��*&
$
Environment/Cumulative Reward��sA���,2       $V�	K	8_˧�A��*#
!
Environment/Episode Length  �Ba ��)       7�_ 	�
8_˧�A��*

Losses/Value LossѦ�@wch�*       ����	t8_˧�A��*

Losses/Policy Lossn��<�;6,       ���E	�%8_˧�A��*

Policy/Learning Rate9�x9��7       ���Y	��Ģ�A��*(
&
Policy/Extrinsic Value Estimatewo@'a��&       sO� 	���Ģ�A��*

Policy/Entropy���?O�J5       ��]�	���Ģ�A��*&
$
Environment/Cumulative RewardPP`Av��2       $V�	��Ģ�A��*#
!
Environment/Episode Length  �B[?`/       m]P	O��Ģ�A��* 

Policy/Extrinsic Reward�laA-%�?)       7�_ 	���Ģ�A��*

Losses/Value Loss�A��l�*       ����	e��Ģ�A��*

Losses/Policy Loss�ȏ<)HWY,       ���E	���Ģ�A��*

Policy/Learning Rate��q9o���7       ���Y	���2ͧ�A��*(
&
Policy/Extrinsic Value Estimate !@�6�/       m]P	��2ͧ�A��* 

Policy/Extrinsic RewardO"�A/�G�&       sO� 	��2ͧ�A��*

Policy/EntropyW��?�p�@5       ��]�	���2ͧ�A��*&
$
Environment/Cumulative Reward�A�8�2       $V�	���2ͧ�A��*#
!
Environment/Episode Length  �B�s�)       7�_ 	5��2ͧ�A��*

Losses/Value Loss�Խ@��b*       ����	���2ͧ�A��*

Losses/Policy LossfX�<|;$�,       ���E	o��2ͧ�A��*

Policy/Learning Rate�Lk9�1Ğ7       ���Y	k�CΧ�A��*(
&
Policy/Extrinsic Value Estimate>�@W�S/       m]P	� DΧ�A��* 

Policy/Extrinsic Reward���A���&       sO� 	l$DΧ�A��*

Policy/Entropy:Ҷ?�8w85       ��]�	�&DΧ�A��*&
$
Environment/Cumulative RewardU��ADgB2       $V�	�(DΧ�A��*#
!
Environment/Episode Length  �B�7��)       7�_ 	+DΧ�A��*

Losses/Value Losst �@��}�*       ����	-DΧ�A��*

Losses/Policy Loss�E�<���g,       ���E	M/DΧ�A��*

Policy/Learning RateB�d9��J7       ���Y	��)!ϧ�A�	*(
&
Policy/Extrinsic Value Estimate�~)@n��&       sO� 	��)!ϧ�A�	*

Policy/Entropy2�??��5       ��]�	�)!ϧ�A�	*&
$
Environment/Cumulative Reward��YAA��2       $V�	��)!ϧ�A�	*#
!
Environment/Episode Length  �B!�Ɠ/       m]P	
�)!ϧ�A�	* 

Policy/Extrinsic RewardGXA�.�)       7�_ 	b�)!ϧ�A�	*

Losses/Value Loss3_�@����*       ����	��)!ϧ�A�	*

Losses/Policy LossB�<x�=_,       ���E	��)!ϧ�A�	*

Policy/Learning Rate�^9_�Y7       ���Y	d=Ч�A��	*(
&
Policy/Extrinsic Value Estimatey<4@şԹ/       m]P	�7=Ч�A��	* 

Policy/Extrinsic Reward�,oA���&       sO� 	�:=Ч�A��	*

Policy/Entropy5�?׎�X5       ��]�	z<=Ч�A��	*&
$
Environment/Cumulative Reward��lA!|��2       $V�	W>=Ч�A��	*#
!
Environment/Episode Length  �B�=-)       7�_ 	�@=Ч�A��	*

Losses/Value Loss��@����*       ����	|B=Ч�A��	*

Losses/Policy Loss���<��
�,       ���E	E=Ч�A��	*

Policy/Learning Rate�mW9����7       ���Y	��bLѧ�A��
*(
&
Policy/Extrinsic Value Estimate��;@6�UR/       m]P	ܹbLѧ�A��
* 

Policy/Extrinsic RewardnnnA	WO&       sO� 	��bLѧ�A��
*

Policy/Entropy�Զ?<���5       ��]�	ϿbLѧ�A��
*&
$
Environment/Cumulative RewardUUoA��H�2       $V�	��bLѧ�A��
*#
!
Environment/Episode Length  �BVN@�)       7�_ 	[�bLѧ�A��
*

Losses/Value Loss��@>$n*       ����	��bLѧ�A��
*

Losses/Policy Loss��<j���,       ���E	?�bLѧ�A��
*

Policy/Learning Rate]�P9Gi�7       ���Y	���Xҧ�A��
*(
&
Policy/Extrinsic Value EstimateO=B@��&       sO� 	�&�Xҧ�A��
*

Policy/Entropyb��?J�m�5       ��]�	]8�Xҧ�A��
*&
$
Environment/Cumulative Reward77OAD��2       $V�	<�Xҧ�A��
*#
!
Environment/Episode Length  �Bߞ�/       m]P	>�Xҧ�A��
* 

Policy/Extrinsic Reward�(PA͌�)       7�_ 	�?�Xҧ�A��
*

Losses/Value Loss˼�@��U*       ����	�A�Xҧ�A��
*

Losses/Policy Loss�6�<f&�,       ���E	�S�Xҧ�A��
*

Policy/Learning Rate�7J9�D�7       ���Y	�?�cӧ�A��*(
&
Policy/Extrinsic Value Estimater=F@ޠ޶/       m]P	�i�cӧ�A��* 

Policy/Extrinsic Reward�'qA���&       sO� 	nl�cӧ�A��*

Policy/Entropy̼�?G�5       ��]�	Xn�cӧ�A��*&
$
Environment/Cumulative Reward��kA�"�2       $V�	p�cӧ�A��*#
!
Environment/Episode Length  �B����)       7�_ 	�q�cӧ�A��*

Losses/Value Loss��@�k��*       ����	�~�cӧ�A��*

Losses/Policy Loss}��<"Ă!,       ���E	1��cӧ�A��*

Policy/Learning Rate�C9Bb��7       ���Y	n1�kԧ�A��*(
&
Policy/Extrinsic Value Estimate�2B@��ܦ/       m]P	Fw�kԧ�A��* 

Policy/Extrinsic Reward��VA��R�&       sO� 	:{�kԧ�A��*

Policy/Entropy���?O/5       ��]�	�kԧ�A��*&
$
Environment/Cumulative RewardUU\A'z?�2       $V�	��kԧ�A��*#
!
Environment/Episode Length  �B��)       7�_ 	<��kԧ�A��*

Losses/Value Loss+�@�0dU*       ����	Y��kԧ�A��*

Losses/Policy Loss�!�<�=D ,       ���E	r��kԧ�A��*

Policy/Learning Rated=9�$'@7       ���Y	�e��ԧ�A��*(
&
Policy/Extrinsic Value Estimate�+A@�h��&       sO� 	����ԧ�A��*

Policy/Entropy���?bg��5       ��]�	����ԧ�A��*&
$
Environment/Cumulative Reward\A����2       $V�	ü��ԧ�A��*#
!
Environment/Episode Length  �B-�
O/       m]P	����ԧ�A��* 

Policy/Extrinsic Reward�N]AȞ�'7       ���Y	}�1�է�A�*(
&
Policy/Extrinsic Value Estimate�;@�c�L/       m]P	J2�է�A�* 

Policy/Extrinsic Reward_LA�c9�)       7�_ 	�2�է�A�*

Losses/Value LossÀ@)QU�*       ����	�#2�է�A�*

Losses/Policy Loss�h�<�qL�,       ���E	�'2�է�A�*

Policy/Learning Rate�s69�ok&       sO� 	�.2�է�A�*

Policy/Entropy���?����5       ��]�	�32�է�A�*&
$
Environment/Cumulative Reward

JA�>v2       $V�	J2�է�A�*#
!
Environment/Episode Length  �B�D7       ���Y	�Is�֧�A��*(
&
Policy/Extrinsic Value EstimatexF<@�L�/       m]P	��s�֧�A��* 

Policy/Extrinsic RewardAAiA�6�u&       sO� 	��s�֧�A��*

Policy/EntropyR��?	�G�)       7�_ 	s�s�֧�A��*

Losses/Value Loss{��@���*       ����	�	t�֧�A��*

Losses/Policy Loss'��<��Tc,       ���E	�t�֧�A��*

Policy/Learning Rate�/9)�i�5       ��]�	Dt�֧�A��*&
$
Environment/Cumulative Reward��nA�	H]2       $V�	wt�֧�A��*#
!
Environment/Episode Length  �B�
Wr7       ���Y	mV6�ק�A��*(
&
Policy/Extrinsic Value Estimate�2@��-&       sO� 	��6�ק�A��*

Policy/Entropyg��?Zp�>5       ��]�	��6�ק�A��*&
$
Environment/Cumulative Reward<<4A/��2       $V�	��6�ק�A��*#
!
Environment/Episode Length  �B�hr�/       m]P	��6�ק�A��* 

Policy/Extrinsic Reward�3A�$b)       7�_ 	��6�ק�A��*

Losses/Value Losspd�@,��(*       ����	�6�ק�A��*

Losses/Policy LossG��<�bIC,       ���E	��6�ק�A��*

Policy/Learning Rate�=)9\[7       ���Y	�~a�ا�A��*(
&
Policy/Extrinsic Value Estimate��F@���/       m]P	�Mb�ا�A��* 

Policy/Extrinsic Reward*�uA�ut&       sO� 	2Sb�ا�A��*

Policy/Entropy�g�?�Z>}5       ��]�	Xb�ا�A��*&
$
Environment/Cumulative Reward<<tA�j�2       $V�	�fb�ا�A��*#
!
Environment/Episode Length  �B�hF�)       7�_ 	wb�ا�A��*

Losses/Value Loss�D@i��1*       ����	{b�ا�A��*

Losses/Policy Loss���<c9; ,       ���E	[}b�ا�A��*

Policy/Learning RateՔ"9�,�7       ���Y	�A)٧�A��*(
&
Policy/Extrinsic Value Estimatec=@Ci�/       m]P	)٧�A��* 

Policy/Extrinsic Reward��KA_*&       sO� 	F�)٧�A��*

Policy/Entropyh�?�c�5       ��]�	P�)٧�A��*&
$
Environment/Cumulative Reward �JA��42       $V�	*�)٧�A��*#
!
Environment/Episode Length  �B��eG)       7�_ 	C�)٧�A��*

Losses/Value Loss2�@����*       ����	��)٧�A��*

Losses/Policy Lossԩ�<�~s,       ���E	׾)٧�A��*

Policy/Learning Rate69p�<�7       ���Y	Pj�lڧ�A��*(
&
Policy/Extrinsic Value Estimate��D@g6k�&       sO� 	w��lڧ�A��*

Policy/Entropyrl�?��,>5       ��]�	��lڧ�A��*&
$
Environment/Cumulative RewardZZbA�Pge2       $V�	n��lڧ�A��*#
!
Environment/Episode Length  �BIa�l/       m]P	8��lڧ�A��* 

Policy/Extrinsic Reward�y`AVrBg)       7�_ 	���lڧ�A��*

Losses/Value Loss�v`@UH��*       ����	���lڧ�A��*

Losses/Policy Losswٳ<��(,       ���E	z��lڧ�A��*

Policy/Learning Rate�^93^7       ���Y	��dXۧ�A��*(
&
Policy/Extrinsic Value Estimate?YG@���/       m]P	3�dXۧ�A��* 

Policy/Extrinsic RewardW?�Af!��&       sO� 	� eXۧ�A��*

Policy/Entropy#\�?�U��5       ��]�	eXۧ�A��*&
$
Environment/Cumulative Reward���A��K�2       $V�	� eXۧ�A��*#
!
Environment/Episode Length  �B�@B�)       7�_ 	d$eXۧ�A��*

Losses/Value Loss<��@�w�*       ����	5eXۧ�A��*

Losses/Policy Loss�}�<�cG�,       ���E	�8eXۧ�A��*

Policy/Learning Rate��9f,P�7       ���Y	�7Eܧ�A��*(
&
Policy/Extrinsic Value Estimatev�@@��/       m]P	b�Eܧ�A��* 

Policy/Extrinsic Reward��VA��&&       sO� 	<�Eܧ�A��*

Policy/Entropy�Z�?�Y��5       ��]�	�Eܧ�A��*&
$
Environment/Cumulative RewardUUYA�7L2       $V�	��Eܧ�A��*#
!
Environment/Episode Length  �B�j>)       7�_ 	��Eܧ�A��*

Losses/Value Loss�˜@����*       ����	�Eܧ�A��*

Losses/Policy Loss�5�<WUB
,       ���E	��Eܧ�A��*

Policy/Learning RateA(9��7       ���Y	��6ݧ�A�*(
&
Policy/Extrinsic Value Estimate/1@l��&       sO� 	:��6ݧ�A�*

Policy/Entropy�[�?ϟ�5       ��]�	���6ݧ�A�*&
$
Environment/Cumulative RewardFF6AIX@]2       $V�	C�6ݧ�A�*#
!
Environment/Episode Length  �B�Q'/       m]P	�6ݧ�A�* 

Policy/Extrinsic Rewarde�6A����)       7�_ 	�'�6ݧ�A�*

Losses/Value Lossq��@�*       ����	�-�6ݧ�A�*

Losses/Policy Loss�i�<�;��,       ���E	*4�6ݧ�A�*

Policy/Learning Rate��9���7       ���Y	F@T%ާ�A��*(
&
Policy/Extrinsic Value Estimate��;@�z�/       m]P	�\T%ާ�A��* 

Policy/Extrinsic Reward2kA�oe&       sO� 	W_T%ާ�A��*

Policy/Entropyl�?A�i�5       ��]�	XmT%ާ�A��*&
$
Environment/Cumulative Reward--uAL��2       $V�	TqT%ާ�A��*#
!
Environment/Episode Length  �B>P��)       7�_ 	SsT%ާ�A��*

Losses/Value Loss��-@ԘS>*       ����	uT%ާ�A��*

Losses/Policy LossU׿<���1,       ���E	�vT%ާ�A��*

Policy/Learning Rate���8C���7       ���Y	��ߧ�A��*(
&
Policy/Extrinsic Value Estimate�,@<@s�/       m]P	#ߧ�A��* 

Policy/Extrinsic Reward  @A�M�&       sO� 	Lߧ�A��*

Policy/Entropy[}�?��T5       ��]�	aߧ�A��*&
$
Environment/Cumulative RewardU�3A~�Z	2       $V�	�ߧ�A��*#
!
Environment/Episode Length  �BL:Q)       7�_ 	J!ߧ�A��*

Losses/Value Loss�3�@%��l*       ����	B#ߧ�A��*

Losses/Policy Lossy��<�w��,       ���E	�0ߧ�A��*

Policy/Learning Rate���8�?��7       ���Y	����ߧ�A��*(
&
Policy/Extrinsic Value EstimateT�B@7`��&       sO� 	����ߧ�A��*

Policy/Entropy�z�?��;�5       ��]�	%���ߧ�A��*&
$
Environment/Cumulative Reward}}}A�ŹX2       $V�	����ߧ�A��*#
!
Environment/Episode Length  �Bl��/       m]P	S���ߧ�A��* 

Policy/Extrinsic Reward�A�Z��)       7�_ 	����ߧ�A��*

Losses/Value Loss=�~@�p�3*       ����	����ߧ�A��*

Losses/Policy LossA��<G���,       ���E	Z���ߧ�A��*

Policy/Learning Rate^w�8�	��7       ���Y	�����A��*(
&
Policy/Extrinsic Value Estimatev�=@��+/       m]P	�����A��* 

Policy/Extrinsic Rewardq�sA�54D&       sO� 	K����A��*

Policy/Entropyum�?~+��5       ��]�	�����A��*&
$
Environment/Cumulative Reward��nAq�S�2       $V�	@����A��*#
!
Environment/Episode Length  �B-5`�)       7�_ 	P����A��*

Losses/Value Loss!�@^X(*       ����	C����A��*

Losses/Policy Loss2��<��Q,       ���E	�����A��*

Policy/Learning Rate#\�8�'�7       ���Y	4�R���A��*(
&
Policy/Extrinsic Value Estimate�)@��z�/       m]P	H�R���A��* 

Policy/Extrinsic Reward��LAd��m&       sO� 	��R���A��*

Policy/Entropy�f�?0�5       ��]�	��R���A��*&
$
Environment/Cumulative RewardUUQA��82       $V�	g�R���A��*#
!
Environment/Episode Length  �B���)       7�_ 	�R���A��*

Losses/Value Loss�]�@M#��*       ����	�R���A��*

Losses/Policy Loss���<��z?,       ���E	{�R���A��*

Policy/Learning Rate�
�8%Ne7       ���Y	隮���A��*(
&
Policy/Extrinsic Value Estimateѻ?@Pzh�&       sO� 	�ܮ���A��*

Policy/Entropy�g�?)dа5       ��]�	�����A��*&
$
Environment/Cumulative Reward�At��2       $V�	,����A��*#
!
Environment/Episode Length  �B�Ǔ�/       m]P	������A��* 

Policy/Extrinsic Reward[��Aъ�)       7�_ 	����A��*

Losses/Value Loss֪@x-*       ����	�����A��*

Losses/Policy Loss��<4[W,       ���E	�����A��*

Policy/Learning Rate��8T3�7       ���Y	�����A��*(
&
Policy/Extrinsic Value Estimate�|2@��B/       m]P	�����A��* 

Policy/Extrinsic Reward��ZA0��{&       sO� 	�����A��*

Policy/Entropyl[�?�l/5       ��]�	������A��*&
$
Environment/Cumulative Reward

RA
o��2       $V�	������A��*#
!
Environment/Episode Length  �B���)       7�_ 	 ����A��*

Losses/Value Loss
��@�o�6*       ����	����A��*

Losses/Policy Loss���<̡�^,       ���E	�����A��*

Policy/Learning Rate;��8���^7       ���Y	��&���A��*(
&
Policy/Extrinsic Value Estimate��3@GZ�/       m]P	�&���A��* 

Policy/Extrinsic Reward��^A.;�&       sO� 	�&���A��*

Policy/Entropy�W�?��'5       ��]�	��&���A��*&
$
Environment/Cumulative Reward �fA0��2       $V�	G�&���A��*#
!
Environment/Episode Length  �B'��)       7�_ 	�&���A��*

Losses/Value Loss@.�_-*       ����	��&���A��*

Losses/Policy Loss��<�%�H,       ���E	��&���A��*

Policy/Learning Rate ��8U�>�7       ���Y	�*ŏ��A��*(
&
Policy/Extrinsic Value Estimate��@]ܹ�&       sO� 	;ŏ��A��*

Policy/Entropy�V�?�X�5       ��]�	=ŏ��A��*&
$
Environment/Cumulative Reward��IA��2       $V�	�>ŏ��A��*#
!
Environment/Episode Length  �B�\y�/       m]P	y@ŏ��A��* 

Policy/Extrinsic Reward�XIA�(X�)       7�_ 	Bŏ��A��*

Losses/Value LossƧ@�~R[*       ����	�Cŏ��A��*

Losses/Policy Loss�
�<!A�m,       ���E	FFŏ��A��*

Policy/Learning Rate�1�8��Z7       ���Y	>��y��A��*(
&
Policy/Extrinsic Value Estimate�6@���n/       m]P	G��y��A��* 

Policy/Extrinsic RewardR�oAŭ&       sO� 	 ��y��A��*

Policy/Entropym�?=/��5       ��]�	>��y��A��*&
$
Environment/Cumulative RewardPPxA�d�2       $V�	��y��A��*#
!
Environment/Episode Length  �BH��)       7�_ 	ẏy��A��*

Losses/Value LossA4�@a��>*       ����	�Їy��A��*

Losses/Policy LossiA�<1J�,       ���E	�҇y��A��*

Policy/Learning Rate�,~8�NS�7       ���Y	ۉm��A��*(
&
Policy/Extrinsic Value Estimatev]3@X�/       m]P	W�m��A��* 

Policy/Extrinsic Reward��bAΙj�&       sO� 	x�m��A��*

Policy/Entropy�z�?���5       ��]�	_�m��A��*&
$
Environment/Cumulative Reward  YA�\2       $V�	�m��A��*#
!
Environment/Episode Length  �B#v�)       7�_ 	��m��A��*

Losses/Value Loss{2�@�&>*       ����	��m��A��*

Losses/Policy LossG��<%�T�,       ���E	o�m��A��*

Policy/Learning Rate)�c8�\�7       ���Y	�ѫ��A��*(
&
Policy/Extrinsic Value EstimateWc5@��o�&       sO� 	�0ѫ��A��*

Policy/Entropynu�?.�z>5       ��]�	L4ѫ��A��*&
$
Environment/Cumulative Reward��pAʏ62       $V�	�6ѫ��A��*#
!
Environment/Episode Length  �B�K/       m]P	�8ѫ��A��* 

Policy/Extrinsic Reward�7rA���7       ���Y	����A��*(
&
Policy/Extrinsic Value EstimateH'1@%"
`/       m]P	�I���A��* 

Policy/Extrinsic Reward��rA�=5)       7�_ 	�N���A��*

Losses/Value Loss1��@�&�A*       ����	�^���A��*

Losses/Policy Loss��<ׂe�,       ���E	|b���A��*

Policy/Learning Rate�SI8_
��&       sO� 	�d���A��*

Policy/Entropy_^�?�s��5       ��]�	�f���A��*&
$
Environment/Cumulative RewardPPpAI�P2       $V�	Dh���A��*#
!
Environment/Episode Length  �Bi�y7       ���Y	��W���A��*(
&
Policy/Extrinsic Value Estimate��3@J?�/       m]P	\;X���A��* 

Policy/Extrinsic Reward��vA�<��&       sO� 	6>X���A��*

Policy/EntropyFT�?gf#�)       7�_ 	@X���A��*

Losses/Value Loss���@�
�	*       ����	�AX���A��*

Losses/Policy Losso�=?�pV,       ���E	�OX���A��*

Policy/Learning Rate�.8�̇|5       ��]�	LSX���A��*&
$
Environment/Cumulative Reward�*{A�y�%2       $V�	MkX���A��*#
!
Environment/Episode Length  �B��@�7       ���Y	��m��Aл*(
&
Policy/Extrinsic Value Estimate��-@m夰&       sO� 	�hm��Aл*

Policy/Entropy�S�??+5       ��]�	�lm��Aл*&
$
Environment/Cumulative Reward}}mA�96�2       $V�	�nm��Aл*#
!
Environment/Episode Length  �B����/       m]P		qm��Aл* 

Policy/Extrinsic RewardujnA�!Ȥ)       7�_ 	sm��Aл*

Losses/Value Loss���@]Z��*       ����	um��Aл*

Losses/Policy Loss5�=�֦�,       ���E	�wm��Aл*

Policy/Learning Rate�z8�+�7       ���Y	��xX��A��*(
&
Policy/Extrinsic Value Estimate�Z)@��/       m]P	b�xX��A��* 

Policy/Extrinsic Reward�YnA�֡&       sO� 	r�xX��A��*

Policy/Entropy O�?�6x5       ��]�	�yX��A��*&
$
Environment/Cumulative RewardgA�9�2       $V�	�yX��A��*#
!
Environment/Episode Length  �B����)       7�_ 	�yX��A��*

Losses/Value LossW"�@p�y�*       ����	�yX��A��*

Losses/Policy Loss��<l��r,       ���E	ZyX��A��*

Policy/Learning Rate���7�9]7       ���Y	��R��A��*(
&
Policy/Extrinsic Value Estimate7 @@��/       m]P	\w�R��A��* 

Policy/Extrinsic Reward���A���&       sO� 	1z�R��A��*

Policy/Entropy�K�?i�5       ��]�	���R��A��*&
$
Environment/Cumulative Reward���A�n�2       $V�	���R��A��*#
!
Environment/Episode Length  �BbP�)       7�_ 	���R��A��*

Losses/Value Loss2�@r�$*       ����	���R��A��*

Losses/Policy Loss
{�< �ud,       ���E	W��R��A��*

Policy/Learning Rate�B�7���7       ���Y	�ӌD���A��*(
&
Policy/Extrinsic Value Estimate*%@2�[&       sO� 	N'�D���A��*

Policy/Entropy�I�?TT�5       ��]�	+�D���A��*&
$
Environment/Cumulative RewardsscA��,n2       $V�	f-�D���A��*#
!
Environment/Episode Length  �B�P1/       m]P	�/�D���A��* 

Policy/Extrinsic Reward5bA;#j)       7�_ 	�3�D���A��*

Losses/Value Lossz<�@���*       ����	jN�D���A��*

Losses/Policy Loss��<%�=�,       ���E	�R�D���A��*

Policy/Learning Rateq��7��7       ���Y	�m�2��A��*(
&
Policy/Extrinsic Value Estimate�$@@�RC�/       m]P	���2��A��* 

Policy/Extrinsic Reward%2�A�~�&       sO� 	���2��A��*

Policy/EntropyNJ�?U\#�5       ��]�	���2��A��*&
$
Environment/Cumulative Rewardnn�AV�(�2       $V�	��2��A��*#
!
Environment/Episode Length  �BR��)       7�_ 	"��2��A��*

Losses/Value Loss_�@�W��*       ����		�2��A��*

Losses/Policy LossI��<��5,       ���E	��2��A��*

Policy/Learning Rate!+7��*a7       ���Y	����A��*(
&
Policy/Extrinsic Value Estimate�.1@�{�/       m]P	�O���A��* 

Policy/Extrinsic RewardUUuAߜ�<&       sO� 	in���A��*

Policy/Entropy�J�?�v?5       ��]�	r���A��*&
$
Environment/Cumulative RewardU�xA:o�;2       $V�	t���A��*#
!
Environment/Episode Length  �B�o��)       7�_ 	j����A��*

Losses/Value LossJ�@$o�%*       ����	�����A��*

Losses/Policy LossV�<��_�,       ���E	e����A��*

Policy/Learning Rate�,�6@���