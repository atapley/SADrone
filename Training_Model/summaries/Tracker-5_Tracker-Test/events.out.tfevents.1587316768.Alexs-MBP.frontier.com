       �K"	   "��Abrain.Event:2N,�lF      n՚R	�$$"��A*�
�
HyperparametersB�BtrainerBppoB
batch_sizeB1024BbetaB0.005Bbuffer_sizeB10240BepsilonB0.2Bhidden_unitsB128BlambdB0.95Blearning_rateB0.0003Blearning_rate_scheduleBlinearB	max_stepsB5.0e5Bmemory_sizeB128B	normalizeBFalseB	num_epochB3B
num_layersB2Btime_horizonB64Bsequence_lengthB64Bsummary_freqB10000Buse_recurrentBFalseBvis_encode_typeBsimpleBreward_signalsB/{'extrinsic': {'strength': 1.0, 'gamma': 0.99}}Bsummary_pathBTracker-5_Tracker-TestB
model_pathB./models/Tracker-5/Tracker-TestBkeep_checkpointsB5J

text����%       �6�	��L"��A�N*

Policy/Entropyş�?FN�}6       OW��	�=�L"��A�N*(
&
Policy/Extrinsic Value EstimateI���$4       ^3\	B�L"��A�N*&
$
Environment/Cumulative Reward��(AA@�M1       ����	
K�L"��A�N*#
!
Environment/Episode Length  �B�*.       ��W�	HO�L"��A�N* 

Policy/Extrinsic Reward  +A�6�7       ���Y	k�8,#��A��*(
&
Policy/Extrinsic Value Estimate�5)><�+/       m]P	�g9,#��A��* 

Policy/Extrinsic Reward--A��L�&       sO� 	Yk9,#��A��*

Policy/Entropy���?^!K�)       7�_ 	n9,#��A��*

Losses/Value Loss�@���8*       ����	�p9,#��A��*

Losses/Policy Loss3��<�k�,       ���E	�r9,#��A��*

Policy/Learning Rate���9o��5       ��]�	�x9,#��A��*&
$
Environment/Cumulative RewardU�Au�sN2       $V�	({9,#��A��*#
!
Environment/Episode Length  �B>�Qk7       ���Y	�[$��A��*(
&
Policy/Extrinsic Value Estimate��>f>g8&       sO� 	�[$��A��*

Policy/Entropy��?��X5       ��]�		�[$��A��*&
$
Environment/Cumulative Reward.AZ92       $V�	:�[$��A��*#
!
Environment/Episode Length  �BsKjo/       m]P	L \$��A��* 

Policy/Extrinsic Rewarduj.A̟�)       7�_ 	O\$��A��*

Losses/Value Loss
�V@���*       ����	K!\$��A��*

Losses/Policy Loss�%�<�\],       ���E	%\$��A��*

Policy/Learning Rate-��9���7       ���Y	�w#�$��A��*(
&
Policy/Extrinsic Value Estimate�X?��/       m]P	��#�$��A��* 

Policy/Extrinsic Reward��A$� �&       sO� 	��#�$��A��*

Policy/Entropy[��?T�5       ��]�	Й#�$��A��*&
$
Environment/Cumulative Reward��AP�_J2       $V�	��#�$��A��*#
!
Environment/Episode Length  �B���)       7�_ 	�#�$��A��*

Losses/Value LossIY@!��*       ����	��#�$��A��*

Losses/Policy Loss�<�<�X�Y,       ���E	$�#�$��A��*

Policy/Learning Rate�Y�9����7       ���Y		N
�%��AІ*(
&
Policy/Extrinsic Value EstimateG�N?Y!��/       m]P	"�
�%��AІ* 

Policy/Extrinsic RewardAA!A��L�&       sO� 	��
�%��AІ*

Policy/Entropy�;�?V�I�5       ��]�	��
�%��AІ*&
$
Environment/Cumulative Reward  A�� 72       $V�	��
�%��AІ*#
!
Environment/Episode Length  �B����)       7�_ 	�
�%��AІ*

Losses/Value Loss%Q:@IC��*       ����	>�
�%��AІ*

Losses/Policy Loss16�<9�3�,       ���E	��
�%��AІ*

Policy/Learning Rate
�9�as7       ���Y	�Dh�&��A��*(
&
Policy/Extrinsic Value Estimate�?�$��&       sO� 	zh�&��A��*

Policy/Entropy��?Vz�5       ��]�	;�h�&��A��*&
$
Environment/Cumulative RewardZZ"A3�2       $V�	ٕh�&��A��*#
!
Environment/Episode Length  �B�K%�/       m]P	�h�&��A��* 

Policy/Extrinsic Reward�S#A�m;=)       7�_ 	$�h�&��A��*

Losses/Value Loss��H@��&*       ����	$�h�&��A��*

Losses/Policy Loss�E�<����,       ���E	��h�&��A��*

Policy/Learning Rate���9	��7       ���Y	噣�'��A�*(
&
Policy/Extrinsic Value Estimateqg�? Bv /       m]P	��'��A�* 

Policy/Extrinsic Rewardq�3Af��4&       sO� 	���'��A�*

Policy/Entropy�:�?M!�b5       ��]�	2��'��A�*&
$
Environment/Cumulative Rewardii1A"tG32       $V�	�-��'��A�*#
!
Environment/Episode Length  �B�hz)       7�_ 	n1��'��A�*

Losses/Value Lossmz@�%�S*       ����	i3��'��A�*

Losses/Policy Loss��<N�	8,       ���E	35��'��A�*

Policy/Learning Rate�w�9���7       ���Y	;��q(��A��*(
&
Policy/Extrinsic Value Estimate�ƺ?+�qO/       m]P	j1�q(��A��* 

Policy/Extrinsic Reward

A��!�&       sO� 	�@�q(��A��*

Policy/Entropy�U�?��^5       ��]�	�E�q(��A��*&
$
Environment/Cumulative RewardUUA�8K>2       $V�	�T�q(��A��*#
!
Environment/Episode Length  �Bd�v)       7�_ 	CX�q(��A��*

Losses/Value Loss4�u@����*       ����	-Z�q(��A��*

Losses/Policy Loss���<�X��,       ���E	�[�q(��A��*

Policy/Learning Rate�#�9����7       ���Y	L��T)��A��*(
&
Policy/Extrinsic Value Estimate�<�?2j�&       sO� 	=�T)��A��*

Policy/Entropyy1�?��:y5       ��]�	Y�T)��A��*&
$
Environment/Cumulative Reward<A��۾2       $V�	��T)��A��*#
!
Environment/Episode Length  �B3���/       m]P	��T)��A��* 

Policy/Extrinsic Reward�N=A�,܂)       7�_ 	�#�T)��A��*

Losses/Value LossaT@��j�*       ����	�&�T)��A��*

Losses/Policy Loss#:�<v�},       ���E	4)�T)��A��*

Policy/Learning Rate�܂9i�u	7       ���Y	9_@5*��A��*(
&
Policy/Extrinsic Value Estimate���?n	�/       m]P	�v@5*��A��* 

Policy/Extrinsic Reward��&A�(�&       sO� 	��@5*��A��*

Policy/Entropy��?�,��5       ��]�	r�@5*��A��*&
$
Environment/Cumulative RewardAA1A!�ơ2       $V�	u�@5*��A��*#
!
Environment/Episode Length  �B䓾)       7�_ 	C�@5*��A��*

Losses/Value Loss{p�@3�%�*       ����	��@5*��A��*

Losses/Policy Lossfi�<���,       ���E	��@5*��A��*

Policy/Learning Rate�9�G3�7       ���Y	�+��A��*(
&
Policy/Extrinsic Value Estimate'�?Tm�/       m]P	c�+��A��* 

Policy/Extrinsic Reward)A���&       sO� 	�+��A��*

Policy/Entropy�$�?!4)5       ��]�	A�+��A��*&
$
Environment/Cumulative RewardUUA�)�2       $V�	N�+��A��*#
!
Environment/Episode Length  �B���})       7�_ 	M�+��A��*

Losses/Value Loss|�t@��
.*       ����	D�+��A��*

Losses/Policy Loss���<�W��,       ���E	8�+��A��*

Policy/Learning Rate9�x9���7       ���Y	T�+��A��*(
&
Policy/Extrinsic Value Estimate %@����&       sO� 	j/�+��A��*

Policy/Entropy�2�?Qڸ75       ��]�	/2�+��A��*&
$
Environment/Cumulative Reward###Ah��2       $V�	]4�+��A��*#
!
Environment/Episode Length  �Bhak�/       m]P	�9�+��A��* 

Policy/Extrinsic Reward�( AВ�)       7�_ 	^L�+��A��*

Losses/Value Lossc7M@��d*       ����	P�+��A��*

Losses/Policy LossU��<��M�,       ���E	DR�+��A��*

Policy/Learning Rate��q9���)7       ���Y	2���,��A��*(
&
Policy/Extrinsic Value EstimateY�@q��/       m]P	����,��A��* 

Policy/Extrinsic Rewardd&As���&       sO� 	s���,��A��*

Policy/Entropy�?��Ī5       ��]�	+���,��A��*&
$
Environment/Cumulative Rewardss#A�c^�2       $V�	˽��,��A��*#
!
Environment/Episode Length  �B`���)       7�_ 	b���,��A��*

Losses/Value Loss��`@B�c*       ����	����,��A��*

Losses/Policy Loss���<�dR,       ���E	�Á�,��A��*

Policy/Learning Rate�Lk9U�H�7       ���Y	���-��A��*(
&
Policy/Extrinsic Value Estimate�@5#!�/       m]P	�k��-��A��* 

Policy/Extrinsic RewardPP8A�.D�&       sO� 		p��-��A��*

Policy/Entropy��?�B#�5       ��]�	!s��-��A��*&
$
Environment/Cumulative Reward�*9A��,2       $V�	�u��-��A��*#
!
Environment/Episode Length  �B�>�)       7�_ 	y��-��A��*

Losses/Value Loss R@����*       ����	�z��-��A��*

Losses/Policy Loss�N�<|e�l,       ���E	p|��-��A��*

Policy/Learning RateB�d9GZ��7       ���Y	}��.��A�	*(
&
Policy/Extrinsic Value EstimateN(@bI#�&       sO� 	���.��A�	*

Policy/Entropy��?W�t�5       ��]�	���.��A�	*&
$
Environment/Cumulative Reward��CA6]�C2       $V�	2��.��A�	*#
!
Environment/Episode Length  �BL3]/       m]P	���.��A�	* 

Policy/Extrinsic Reward�EA�l)       7�_ 	.��.��A�	*

Losses/Value Lossz�@[�~�*       ����	S��.��A�	*

Losses/Policy Loss�C�<���,       ���E	Q��.��A�	*

Policy/Learning Rate�^9!�YB7       ���Y	�rAe/��A��	*(
&
Policy/Extrinsic Value Estimate�/@��̳/       m]P	��Ae/��A��	* 

Policy/Extrinsic Reward�8A�D�&       sO� 	.�Ae/��A��	*

Policy/Entropy�?:��5       ��]�	�Ae/��A��	*&
$
Environment/Cumulative Reward��5A��-�2       $V�	ɒAe/��A��	*#
!
Environment/Episode Length  �BA̛z)       7�_ 	
�Ae/��A��	*

Losses/Value Loss'��@4��*       ����	�Ae/��A��	*

Losses/Policy Loss�ѿ<1�W�,       ���E	��Ae/��A��	*

Policy/Learning Rate�mW9G8�7       ���Y	�ckE0��A��
*(
&
Policy/Extrinsic Value Estimate�-<@8���/       m]P	��kE0��A��
* 

Policy/Extrinsic RewardnnFA'�`�&       sO� 	G�kE0��A��
*

Policy/Entropys��?�05       ��]�	�kE0��A��
*&
$
Environment/Cumulative Reward �KA�A��2       $V�	��kE0��A��
*#
!
Environment/Episode Length  �BzV[)       7�_ 	N�kE0��A��
*

Losses/Value Loss�@P���*       ����	؟kE0��A��
*

Losses/Policy LossdG�<]:�r,       ���E	�kE0��A��
*

Policy/Learning Rate]�P9�7       ���Y	'�1��A��
*(
&
Policy/Extrinsic Value Estimate��D@�W�&       sO� 	�'�1��A��
*

Policy/Entropy���?Q+&5       ��]�	*�1��A��
*&
$
Environment/Cumulative Reward��ZA��2       $V�	�+�1��A��
*#
!
Environment/Episode Length  �B-_�/       m]P	�-�1��A��
* 

Policy/Extrinsic Reward&�ZA8�)       7�_ 	f/�1��A��
*

Losses/Value Loss(�@��i*       ����	1�1��A��
*

Losses/Policy Loss�n�<f��m,       ���E	�3�1��A��
*

Policy/Learning Rate�7J9�Z7       ���Y	��g�1��A��*(
&
Policy/Extrinsic Value Estimate(�U@r� �/       m]P	Th�1��A��* 

Policy/Extrinsic RewardBG5A�Ǿ�&       sO� 	�,h�1��A��*

Policy/EntropyMt�?T���5       ��]�	�0h�1��A��*&
$
Environment/Cumulative Reward��2A��rE2       $V�	�2h�1��A��*#
!
Environment/Episode Length  �B�J�)       7�_ 	H4h�1��A��*

Losses/Value LossSA]@Kz�*       ����	�5h�1��A��*

Losses/Policy Losss
�<S��,       ���E	�7h�1��A��*

Policy/Learning Rate�C9v�@g7       ���Y	f8�2��A��*(
&
Policy/Extrinsic Value Estimaten�`@uq&9/       m]P	�w8�2��A��* 

Policy/Extrinsic RewardKKCA�t�[&       sO� 	�y8�2��A��*

Policy/Entropy�Y�?SQ:C5       ��]�	�{8�2��A��*&
$
Environment/Cumulative Reward �FAS��!2       $V�	Z}8�2��A��*#
!
Environment/Episode Length  �B+dA�)       7�_ 	�~8�2��A��*

Losses/Value Loss��@�P��*       ����	�8�2��A��*

Losses/Policy Loss�"�<]�  ,       ���E	�8�2��A��*

Policy/Learning Rated=9!=ڂ7       ���Y	� 3��A��*(
&
Policy/Extrinsic Value Estimatem@<��O&       sO� 	� 3��A��*

Policy/Entropy�J�? �@�5       ��]�	1# 3��A��*&
$
Environment/Cumulative Rewardii9AGpA2       $V�	5& 3��A��*#
!
Environment/Episode Length  �BD�/       m]P	�( 3��A��* 

Policy/Extrinsic Reward
L:A\v��7       ���Y	8�4��A�*(
&
Policy/Extrinsic Value Estimate{hr@O�Һ/       m]P	l~�4��A�* 

Policy/Extrinsic Reward��?A��)       7�_ 	��4��A�*

Losses/Value Loss\@A,�#*       ����	��4��A�*

Losses/Policy Loss�&�<W�N�,       ���E	��4��A�*

Policy/Learning Rate�s69��&       sO� 	ٔ�4��A�*

Policy/Entropy�S�?�U��5       ��]�	���4��A�*&
$
Environment/Cumulative Reward��BA�q&2       $V�	9��4��A�*#
!
Environment/Episode Length  �B=��7       ���Y	�s��4��A��*(
&
Policy/Extrinsic Value Estimateb�y@Н��/       m]P	����4��A��* 

Policy/Extrinsic Reward��DA�Rjn&       sO� 	����4��A��*

Policy/Entropy��?qN�w)       7�_ 	E���4��A��*

Losses/Value Loss�Jh@x\�*       ����	���4��A��*

Losses/Policy Loss���<F�g�,       ���E	����4��A��*

Policy/Learning Rate�/9,��;5       ��]�	���4��A��*&
$
Environment/Cumulative Reward �BAx#�Y2       $V�	F���4��A��*#
!
Environment/Episode Length  �B�)17       ���Y	B��5��A��*(
&
Policy/Extrinsic Value EstimateJ]@kS1&       sO� 	�j �5��A��*

Policy/Entropy��?����5       ��]�	Tr �5��A��*&
$
Environment/Cumulative Reward��)A����2       $V�	\v �5��A��*#
!
Environment/Episode Length  �B�Q�X/       m]P	�| �5��A��* 

Policy/Extrinsic Reward�(A�r�)       7�_ 	� �5��A��*

Losses/Value Loss�؀@��[*       ����	{� �5��A��*

Losses/Policy Loss���<���,       ���E	� �5��A��*

Policy/Learning Rate�=)9���7       ���Y	6L�6��A��*(
&
Policy/Extrinsic Value Estimate���@?u^�/       m]P	9FL�6��A��* 

Policy/Extrinsic Rewardb�IA��cv&       sO� 	�dL�6��A��*

Policy/EntropyH�?���.5       ��]�	0hL�6��A��*&
$
Environment/Cumulative Reward��GA�٨/2       $V�	�wL�6��A��*#
!
Environment/Episode Length  �B^�{)       7�_ 	�{L�6��A��*

Losses/Value Loss�$D@��C�*       ����	~L�6��A��*

Losses/Policy Loss�.�<6L��,       ���E	�L�6��A��*

Policy/Learning RateՔ"9 �]7       ���Y	��7��A��*(
&
Policy/Extrinsic Value Estimateٸ�@����/       m]P	�X��7��A��* 

Policy/Extrinsic Reward��IA��<}&       sO� 	]��7��A��*

Policy/Entropy�ٳ?����5       ��]�	f��7��A��*&
$
Environment/Cumulative RewardUUKA8�*2       $V�	�h��7��A��*#
!
Environment/Episode Length  �B�^()       7�_ 	�x��7��A��*

Losses/Value Loss�t@9V�*       ����	5~��7��A��*

Losses/Policy Lossg�<�,       ���E	����7��A��*

Policy/Learning Rate69/�v�7       ���Y	˂a8��A��*(
&
Policy/Extrinsic Value Estimateo�@�Y&       sO� 	���a8��A��*

Policy/Entropy�ݳ?YQ�O5       ��]�	���a8��A��*&
$
Environment/Cumulative RewardPPHA�,E2       $V�	��a8��A��*#
!
Environment/Episode Length  �B�(e /       m]P	&��a8��A��* 

Policy/Extrinsic Reward��CA+	�T)       7�_ 	� �a8��A��*

Losses/Value Loss�M|@��*       ����	��a8��A��*

Losses/Policy Loss�;�<��|,       ���E	��a8��A��*

Policy/Learning Rate�^9mC7       ���Y	�e�?9��A��*(
&
Policy/Extrinsic Value Estimate���@"��/       m]P	4��?9��A��* 

Policy/Extrinsic Reward t4A��P&       sO� 	
��?9��A��*

Policy/Entropy���?����5       ��]�	��?9��A��*&
$
Environment/Cumulative Reward777A�!)�2       $V�	���?9��A��*#
!
Environment/Episode Length  �B���)       7�_ 	���?9��A��*

Losses/Value Loss�@O��*       ����	���?9��A��*

Losses/Policy Loss�n<�,2,       ���E	,��?9��A��*

Policy/Learning Rate��9k�*�7       ���Y	C��:��A��*(
&
Policy/Extrinsic Value Estimate��@���/       m]P	���:��A��* 

Policy/Extrinsic RewardiiAA��&       sO� 	h��:��A��*

Policy/Entropy9��?oN�5       ��]�	̶�:��A��*&
$
Environment/Cumulative Reward�*:AUl�42       $V�	R��:��A��*#
!
Environment/Episode Length  �B���)       7�_ 	���:��A��*

Losses/Value Loss��S@��6^*       ����	���:��A��*

Losses/Policy LossMB�<e`y�,       ���E	���:��A��*

Policy/Learning RateA(9p{�j7       ���Y	j���:��A�*(
&
Policy/Extrinsic Value Estimate�+�@�4�X&       sO� 	j���:��A�*

Policy/Entropy��?�3 5       ��]�	���:��A�*&
$
Environment/Cumulative Reward��MAij��2       $V�	<���:��A�*#
!
Environment/Episode Length  �BV���/       m]P	L���:��A�* 

Policy/Extrinsic Reward#
LA��)       7�_ 	����:��A�*

Losses/Value Loss�&�@O5-_*       ����	����:��A�*

Losses/Policy Loss/6�<W�U�,       ���E	6���:��A�*

Policy/Learning Rate��9���w7       ���Y	.=��;��A��*(
&
Policy/Extrinsic Value Estimate��@�P!�/       m]P	�b��;��A��* 

Policy/Extrinsic Rewardgy2A���9&       sO� 	^f��;��A��*

Policy/Entropyn~�?Y[��5       ��]�	�i��;��A��*&
$
Environment/Cumulative Reward__/A��M�2       $V�	�m��;��A��*#
!
Environment/Episode Length  �B��+�)       7�_ 	R}��;��A��*

Losses/Value Loss��>@��XJ*       ����	����;��A��*

Losses/Policy LossC��<pe\,       ���E	U���;��A��*

Policy/Learning Rate���8`ԥ�7       ���Y	�G�<��A��*(
&
Policy/Extrinsic Value Estimate��@�_�:/       m]P	\n�<��A��* 

Policy/Extrinsic Reward##;ASi��&       sO� 	l��<��A��*

Policy/Entropy�}�?���5       ��]�	���<��A��*&
$
Environment/Cumulative Reward��<A&��2       $V�	���<��A��*#
!
Environment/Episode Length  �B�D��)       7�_ 	���<��A��*

Losses/Value Loss�ϒ@�<�y*       ����	���<��A��*

Losses/Policy Loss'�<��{�,       ���E	���<��A��*

Policy/Learning Rate���8#4�7       ���Y	0�p�=��A��*(
&
Policy/Extrinsic Value Estimate�@�@�L��&       sO� 	��p�=��A��*

Policy/Entropy{{�?����5       ��]�	��p�=��A��*&
$
Environment/Cumulative RewardFA�S-~2       $V�	J�p�=��A��*#
!
Environment/Episode Length  �BZ���/       m]P	_�p�=��A��* 

Policy/Extrinsic Reward��FA*b�)       7�_ 	��p�=��A��*

Losses/Value Loss+�R@�<�*       ����	Y�p�=��A��*

Losses/Policy Lossɿ�<e��=,       ���E	i q�=��A��*

Policy/Learning Rate^w�8����7       ���Y	.7�y>��A��*(
&
Policy/Extrinsic Value Estimateђ�@��!�/       m]P	1[�y>��A��* 

Policy/Extrinsic RewardӰA����&       sO� 	[^�y>��A��*

Policy/Entropyx�?ZR�5       ��]�	h`�y>��A��*&
$
Environment/Cumulative Reward��A���@2       $V�	Fb�y>��A��*#
!
Environment/Episode Length  �B�HCL)       7�_ 	(d�y>��A��*

Losses/Value Losspcc@��,*       ����	�e�y>��A��*

Losses/Policy Loss�ԩ<���p,       ���E	�q�y>��A��*

Policy/Learning Rate#\�8�b��7       ���Y	/�tZ?��A��*(
&
Policy/Extrinsic Value Estimateؒ�@�-T/       m]P	��tZ?��A��* 

Policy/Extrinsic Reward

ZA��B�&       sO� 	��tZ?��A��*

Policy/Entropy���?��5       ��]�	F�tZ?��A��*&
$
Environment/Cumulative Reward  _A��k2       $V�	g�tZ?��A��*#
!
Environment/Episode Length  �B�r'�)       7�_ 	o�tZ?��A��*

Losses/Value Loss�=s@Ac�*       ����	o�tZ?��A��*

Losses/Policy Loss'F�<cj�z,       ���E	b�tZ?��A��*

Policy/Learning Rate�
�8\�w�7       ���Y	9��7@��A��*(
&
Policy/Extrinsic Value Estimatewe�@b��>&       sO� 	���7@��A��*

Policy/Entropy0��?�`�5       ��]�	���7@��A��*&
$
Environment/Cumulative RewardDA����2       $V�	���7@��A��*#
!
Environment/Episode Length  �BZ��/       m]P	���7@��A��* 

Policy/Extrinsic Reward߈BA�l��)       7�_ 	8��7@��A��*

Losses/Value Loss�4�@��o*       ����	A��7@��A��*

Losses/Policy Loss��<M�(,       ���E	���7@��A��*

Policy/Learning Rate��8z�(7       ���Y	+k�A��A��*(
&
Policy/Extrinsic Value EstimateH�@���/       m]P	�z�A��A��* 

Policy/Extrinsic Reward�AGA���&       sO� 	n��A��A��*

Policy/Entropyf��?���5       ��]�	$��A��A��*&
$
Environment/Cumulative Reward}}EA���2       $V�	ؠ�A��A��*#
!
Environment/Episode Length  �B��v)       7�_ 	��A��A��*

Losses/Value Loss���@�fN%*       ����	�A��A��*

Losses/Policy LossM~�<	�<y,       ���E	���A��A��*

Policy/Learning Rate;��8�A�87       ���Y	�N�A��A��*(
&
Policy/Extrinsic Value EstimateS-�@�Ę�/       m]P	�N�A��A��* 

Policy/Extrinsic RewardxxXA}��t&       sO� 	�N�A��A��*

Policy/Entropy+p�?��95       ��]�	߫N�A��A��*&
$
Environment/Cumulative RewardU�YA�0��2       $V�	��N�A��A��*#
!
Environment/Episode Length  �B�N�)       7�_ 	\�N�A��A��*

Losses/Value Loss�S�@т�	*       ����	��N�A��A��*

Losses/Policy Loss���<:��y,       ���E	h�N�A��A��*

Policy/Learning Rate ��8Z=7       ���Y	����B��A��*(
&
Policy/Extrinsic Value Estimate�ʭ@��E�&       sO� 	���B��A��*

Policy/Entropy�W�?�N�P5       ��]�	 ���B��A��*&
$
Environment/Cumulative RewardiiaAڟ�J2       $V�	����B��A��*#
!
Environment/Episode Length  �B�mJ�/       m]P	6���B��A��* 

Policy/Extrinsic Reward�cA�D,Q)       7�_ 	���B��A��*

Losses/Value Loss=:�@�zИ*       ����	���B��A��*

Losses/Policy Loss|��<G��C,       ���E	���B��A��*

Policy/Learning Rate�1�8�[J�7       ���Y	l�j�C��A��*(
&
Policy/Extrinsic Value Estimate݂�@*�N;/       m]P	mrk�C��A��* 

Policy/Extrinsic Reward�4LA?vO&       sO� 	Sxk�C��A��*

Policy/Entropy�L�?���s5       ��]�	�{k�C��A��*&
$
Environment/Cumulative Reward��LA*�2       $V�	֏k�C��A��*#
!
Environment/Episode Length  �B�P37)       7�_ 	(�k�C��A��*

Losses/Value Lossa8�@�*       ����	��k�C��A��*

Losses/Policy Lossꗢ<Z�*�,       ���E	�k�C��A��*

Policy/Learning Rate�,~8�l�7       ���Y	��ۑD��A��*(
&
Policy/Extrinsic Value Estimate�s�@-�K/       m]P	TܑD��A��* 

Policy/Extrinsic RewardFF^A>�O�&       sO� 	P ܑD��A��*

Policy/Entropy�G�?�D55       ��]�	w,ܑD��A��*&
$
Environment/Cumulative Reward�*`A�֗�2       $V�	<0ܑD��A��*#
!
Environment/Episode Length  �Bd��)       7�_ 	{2ܑD��A��*

Losses/Value Loss���@���>*       ����	?4ܑD��A��*

Losses/Policy Loss���<ƨi�,       ���E	�5ܑD��A��*

Policy/Learning Rate)�c8����7       ���Y	㨶�D��A��*(
&
Policy/Extrinsic Value Estimate#t�@Ũ�&       sO� 	*���D��A��*

Policy/Entropy�F�?P�	E5       ��]�	q���D��A��*&
$
Environment/Cumulative Reward��GA[��y2       $V�	����D��A��*#
!
Environment/Episode Length  �B���W/       m]P	���D��A��* 

Policy/Extrinsic Reward�HA�Ͼ7       ���Y	gf�E��A��*(
&
Policy/Extrinsic Value Estimate2��@�d��/       m]P	ʈ�E��A��* 

Policy/Extrinsic Rewardb�YA8
d�)       7�_ 	���E��A��*

Losses/Value Loss��r@c*       ����	��E��A��*

Losses/Policy Loss���<�ƨ,       ���E	���E��A��*

Policy/Learning Rate�SI8:�F&       sO� 	>��E��A��*

Policy/EntropyZD�?x��5       ��]�	��E��A��*&
$
Environment/Cumulative Reward}}]A\^۰2       $V�	���E��A��*#
!
Environment/Episode Length  �Bu��f7       ���Y	�I]�F��A��*(
&
Policy/Extrinsic Value Estimate>��@���/       m]P	�i]�F��A��* 

Policy/Extrinsic RewardWAa��H&       sO� 	�l]�F��A��*

Policy/Entropyq=�?]�]4)       7�_ 	]�F��A��*

Losses/Value Loss#�@���*       ����	Ã]�F��A��*

Losses/Policy Loss2��<�f �,       ���E	#�]�F��A��*

Policy/Learning Rate�.8;�ι5       ��]�	��]�F��A��*&
$
Environment/Cumulative RewardUUTAx��U2       $V�	Л]�F��A��*#
!
Environment/Episode Length  �BC�P�7       ���Y	+0moG��Aл*(
&
Policy/Extrinsic Value Estimatel��@S"7�&       sO� 	@�moG��Aл*

Policy/Entropy�4�?�q��5       ��]�	�moG��Aл*&
$
Environment/Cumulative Reward((PAd	�+2       $V�	@�moG��Aл*#
!
Environment/Episode Length  �B?��/       m]P	��moG��Aл* 

Policy/Extrinsic Reward�]OA�bH�)       7�_ 	��moG��Aл*

Losses/Value Lossņ@F�c�*       ����	{�moG��Aл*

Losses/Policy Loss�E�<Ѩ=,       ���E	z�moG��Aл*

Policy/Learning Rate�z8�p�7       ���Y	��WH��A��*(
&
Policy/Extrinsic Value Estimate���@E��/       m]P	�6WH��A��* 

Policy/Extrinsic Reward�nFAA�q�&       sO� 	WFWH��A��*

Policy/Entropy+*�?�WSZ5       ��]�	�JWH��A��*&
$
Environment/Cumulative Reward��FA(ne�2       $V�	�LWH��A��*#
!
Environment/Episode Length  �B�b>�)       7�_ 	NWH��A��*

Losses/Value Loss�@SjN*       ����	TPWH��A��*

Losses/Policy Loss��<�w�,       ���E	�`WH��A��*

Policy/Learning Rate���7�%��7       ���Y	�j7I��A��*(
&
Policy/Extrinsic Value EstimateW�@IL�`/       m]P	1@k7I��A��* 

Policy/Extrinsic Reward��5A_dϏ&       sO� 	FCk7I��A��*

Policy/Entropy �?@�z5       ��]�	$Ek7I��A��*&
$
Environment/Cumulative Reward��3A$�i2       $V�	Gk7I��A��*#
!
Environment/Episode Length  �B
�lE)       7�_ 	,fk7I��A��*

Losses/Value Loss���@�l�#*       ����	jk7I��A��*

Losses/Policy Lossgy�<����,       ���E	�zk7I��A��*

Policy/Learning Rate�B�76�}�7       ���Y	MKrJ��A��*(
&
Policy/Extrinsic Value Estimate��@qM�&       sO� 	��rJ��A��*

Policy/Entropy]�?%z��5       ��]�	��rJ��A��*&
$
Environment/Cumulative Reward__7AN2       $V�	��rJ��A��*#
!
Environment/Episode Length  �B@@�j/       m]P	i�rJ��A��* 

Policy/Extrinsic Reward� 7A��)       7�_ 	;�rJ��A��*

Losses/Value LossDh@u2�g*       ����	�rJ��A��*

Losses/Policy Loss�g�<��L�,       ���E	�rJ��A��*

Policy/Learning Rateq��7����7       ���Y	Rc(�J��A��*(
&
Policy/Extrinsic Value EstimateW��@F��/       m]P	�(�J��A��* 

Policy/Extrinsic Rewardy9A���h&       sO� 	��(�J��A��*

Policy/Entropy �?\N��5       ��]�	W�(�J��A��*&
$
Environment/Cumulative Reward}}=A/y�2       $V�	I�(�J��A��*#
!
Environment/Episode Length  �B3Y� )       7�_ 	(�(�J��A��*

Losses/Value Losswd@�p*       ����	�(�J��A��*

Losses/Policy Loss���<Jm-j,       ���E	��(�J��A��*

Policy/Learning Rate!+7����7       ���Y	XY��K��A��*(
&
Policy/Extrinsic Value Estimate��@�� �/       m]P	����K��A��* 

Policy/Extrinsic Rewardxx@A{ ��&       sO� 	&���K��A��*

Policy/Entropy��?��_�5       ��]�	����K��A��*&
$
Environment/Cumulative Reward  <A��`J2       $V�	����K��A��*#
!
Environment/Episode Length  �Bs� )       7�_ 	ݘ��K��A��*

Losses/Value LossW�{@)�H*       ����	����K��A��*

Losses/Policy Loss�x�<�u�,       ���E	����K��A��*

Policy/Learning Rate�,�6b�X�