       �K"	  �M��Abrain.Event:2_���F      n՚R	�~�M��A*�
�
HyperparametersB�BtrainerBppoB
batch_sizeB1024BbetaB0.005Bbuffer_sizeB10240BepsilonB0.2Bhidden_unitsB128BlambdB0.95Blearning_rateB0.0003Blearning_rate_scheduleBlinearB	max_stepsB1.0e7Bmemory_sizeB128B	normalizeBFalseB	num_epochB3B
num_layersB2Btime_horizonB64Bsequence_lengthB64Bsummary_freqB10000Buse_recurrentBFalseBvis_encode_typeBsimpleBreward_signalsB/{'extrinsic': {'strength': 1.0, 'gamma': 0.99}}Bsummary_pathBTracker-8_Tracker-TestB
model_pathB./models/Tracker-8/Tracker-TestBkeep_checkpointsB5J

text	r��%       �6�	������A�N*

Policy/Entropyş�?��06       OW��	E�����A�N*(
&
Policy/Extrinsic Value Estimate���00�4       ^3\	,�����A�N*&
$
Environment/Cumulative Reward��RA��g1       ����	+�����A�N*#
!
Environment/Episode Length  �B��W�.       ��W�	�����A�N* 

Policy/Extrinsic RewardUUXA4���7       ���Y	�2���A��*(
&
Policy/Extrinsic Value EstimateGB�>�a)~/       m]P	��2���A��* 

Policy/Extrinsic Reward��XA��&       sO� 	��2���A��*

Policy/Entropy�ҵ?k���)       7�_ 	��2���A��*

Losses/Value Loss�q�@�^L�*       ����	��2���A��*

Losses/Policy Loss���<�K�J,       ���E	�2���A��*

Policy/Learning Rate��9���5       ��]�	4�2���A��*&
$
Environment/Cumulative Reward��^AX�<^2       $V�	s�2���A��*#
!
Environment/Episode Length  �B,��7       ���Y	ӈ���A��*(
&
Policy/Extrinsic Value Estimate��?�<�&       sO� 	>�����A��*

Policy/EntropyD �?gE��5       ��]�	,�����A��*&
$
Environment/Cumulative RewardxxpA����2       $V�	̝����A��*#
!
Environment/Episode Length  �Bi�`/       m]P	9�����A��* 

Policy/Extrinsic RewardD�iA�d�o)       7�_ 	������A��*

Losses/Value Loss$��@�?��*       ����	����A��*

Losses/Policy Loss�Ӳ<6�h ,       ���E	ŧ����A��*

Policy/Learning Rate���9ԞnN7       ���Y	�7���A��*(
&
Policy/Extrinsic Value Estimate��9?��!/       m]P	e�7���A��* 

Policy/Extrinsic Reward5lgA�&       sO� 	S�7���A��*

Policy/Entropy��?���?5       ��]�	J�7���A��*&
$
Environment/Cumulative Reward��ZA���2       $V�	}�7���A��*#
!
Environment/Episode Length  �BPo��)       7�_ 	N�7���A��*

Losses/Value Loss���@��l*       ����	�7���A��*

Losses/Policy Loss#j�<6S��,       ���E	��7���A��*

Policy/Learning Rate%ʜ9���7       ���Y	�>����AІ*(
&
Policy/Extrinsic Value Estimateͩf?,V��/       m]P	�]����AІ* 

Policy/Extrinsic Reward__WA���&       sO� 	o`����AІ*

Policy/Entropy`F�?�';L5       ��]�	Fb����AІ*&
$
Environment/Cumulative Reward �\A��'�2       $V�	�c����AІ*#
!
Environment/Episode Length  �B.ʳ)       7�_ 	{e����AІ*

Losses/Value Loss}�@'��*       ����	
g����AІ*

Losses/Policy Lossǚ�<��_�,       ���E	�i����AІ*

Policy/Learning Rate6��9/��7       ���Y	w����A��*(
&
Policy/Extrinsic Value Estimate3��?�h:�&       sO� 	�����A��*

Policy/Entropy=��?Gx��5       ��]�	K����A��*&
$
Environment/Cumulative RewardnnnA�˔{2       $V�	%����A��*#
!
Environment/Episode Length  �BT���/       m]P	�����A��* 

Policy/Extrinsic Reward�]oA����)       7�_ 	�	����A��*

Losses/Value Loss�H�@�y̼*       ����	r����A��*

Losses/Policy Loss�t�<+�*�,       ���E	����A��*

Policy/Learning Rate�u�9x��7       ���Y	�( ���A�*(
&
Policy/Extrinsic Value EstimatedR�?���/       m]P	�E ���A�* 

Policy/Extrinsic Reward7�sA-z��&       sO� 	�I ���A�*

Policy/Entropy5��?�گ�5       ��]�	IL ���A�*&
$
Environment/Cumulative RewarduAk$Ɔ2       $V�	N ���A�*#
!
Environment/Episode Length  �B��Y)       7�_ 	�O ���A�*

Losses/Value Loss0!�@O��+*       ����	Q ���A�*

Losses/Policy Lossm��<� I�,       ���E	3S ���A�*

Policy/Learning Rate�K�9���7       ���Y	�@����A��*(
&
Policy/Extrinsic Value Estimate}��?�]��/       m]P	U�����A��* 

Policy/Extrinsic Reward��iAU�&       sO� 	;�����A��*

Policy/Entropy~��?=��A5       ��]�	Y�����A��*&
$
Environment/Cumulative Reward  iA���J2       $V�	
�����A��*#
!
Environment/Episode Length  �BI���)       7�_ 	�����A��*

Losses/Value Loss'��@�U�U*       ����	�����A��*

Losses/Policy Loss�N�<���,       ���E	H�����A��*

Policy/Learning Rate!�9E�A7       ���Y	ۉ����A��*(
&
Policy/Extrinsic Value EstimateI��?2>R&       sO� 	{�����A��*

Policy/Entropy>��?s���5       ��]�	�����A��*&
$
Environment/Cumulative Reward��rAU[��2       $V�	s����A��*#
!
Environment/Episode Length  �B�XwI/       m]P	�����A��* 

Policy/Extrinsic Reward��sA��)       7�_ 	{����A��*

Losses/Value Lossd�@f���*       ����	H����A��*

Losses/Policy Loss�.�<m���,       ���E	����A��*

Policy/Learning Rate��9'X^7       ���Y	Ի���A��*(
&
Policy/Extrinsic Value EstimateB��?�K_/       m]P	7ߜ��A��* 

Policy/Extrinsic RewardӰ]A�6��&       sO� 	����A��*

Policy/Entropy윶?#��15       ��]�	~���A��*&
$
Environment/Cumulative Rewardnn^Aѳ�T2       $V�	���A��*#
!
Environment/Episode Length  �B���})       7�_ 	����A��*

Losses/Value Loss���@�%�*       ����	����A��*

Losses/Policy Loss#�<����,       ���E	���A��*

Policy/Learning Rate{̛9?P ;7       ���Y	��� ��A��*(
&
Policy/Extrinsic Value Estimate��@���/       m]P	��� ��A��* 

Policy/Extrinsic Reward��cA#��&       sO� 	��� ��A��*

Policy/Entropy鉶?I�5       ��]�	'�� ��A��*&
$
Environment/Cumulative Reward�*dA�g�B2       $V�	a�� ��A��*#
!
Environment/Episode Length  �B���)       7�_ 	��� ��A��*

Losses/Value Loss�"�@�$�*       ����	q�� ��A��*

Losses/Policy Loss�P�<��?�,       ���E	>�� ��A��*

Policy/Learning Rate���9�b��7       ���Y	��`.!��A��*(
&
Policy/Extrinsic Value EstimateO�@{��&       sO� 	��`.!��A��*

Policy/Entropymh�?��ɞ5       ��]�	��`.!��A��*&
$
Environment/Cumulative Reward��vA��
�2       $V�	�`.!��A��*#
!
Environment/Episode Length  �B�/�/       m]P	�a.!��A��* 

Policy/Extrinsic Reward�IwA����)       7�_ 	a.!��A��*

Losses/Value Lossi��@qvH�*       ����	Da.!��A��*

Losses/Policy Loss���<w�I�,       ���E	O%a.!��A��*

Policy/Learning Rate�w�9�
Er7       ���Y	��m-"��A��*(
&
Policy/Extrinsic Value Estimate5L@�b/       m]P	�n-"��A��* 

Policy/Extrinsic Reward��A��&       sO� 	Vn-"��A��*

Policy/EntropyT�?��j�5       ��]�	3n-"��A��*&
$
Environment/Cumulative RewardA��22       $V�	�n-"��A��*#
!
Environment/Episode Length  �BsA�)       7�_ 	�n-"��A��*

Losses/Value Loss�k�@�+�*       ����	.n-"��A��*

Losses/Policy Loss�<��,       ���E	�#n-"��A��*

Policy/Learning Rate�M�9�f��7       ���Y	�E�+#��A��*(
&
Policy/Extrinsic Value Estimate n+@x��Z/       m]P	Z֡+#��A��* 

Policy/Extrinsic Reward��dA�M��&       sO� 	��+#��A��*

Policy/Entropy�V�?��h5       ��]�	��+#��A��*&
$
Environment/Cumulative RewardU�hA�i2       $V�	���+#��A��*#
!
Environment/Episode Length  �B�]��)       7�_ 	$�+#��A��*

Losses/Value Loss���@��+*       ����	��+#��A��*

Losses/Policy Losszm�<�u�z,       ���E	���+#��A��*

Policy/Learning Rate]#�9$��X7       ���Y	�!8$��A�	*(
&
Policy/Extrinsic Value Estimate]�5@�;��&       sO� 	p8$��A�	*

Policy/Entropy�\�?S^�5       ��]�		s8$��A�	*&
$
Environment/Cumulative RewardxxHAgA˖2       $V�	Pu8$��A�	*#
!
Environment/Episode Length  �BȲ/       m]P	pw8$��A�	* 

Policy/Extrinsic Reward�XIA ���)       7�_ 	}y8$��A�	*

Losses/Value LossK��@l�t*       ����	ْ8$��A�	*

Losses/Policy Loss��< R��,       ���E	��8$��A�	*

Policy/Learning Ratek��9Ȏ*�7       ���Y	��
%��A��	*(
&
Policy/Extrinsic Value EstimatePl?@
�6/       m]P	��
%��A��	* 

Policy/Extrinsic Reward��rA��hM&       sO� 	��
%��A��	*

Policy/Entropy�f�?��.5       ��]�	��
%��A��	*&
$
Environment/Cumulative Reward��|A�H2�2       $V�	��
%��A��	*#
!
Environment/Episode Length  �B�y�)       7�_ 	��
%��A��	*

Losses/Value Loss���@��(�*       ����	f�
%��A��	*

Losses/Policy Loss�T�<G��,       ���E	��
%��A��	*

Policy/Learning Rate�Κ9�f2U7       ���Y	�a  &��A��
*(
&
Policy/Extrinsic Value Estimate��I@8gͅ/       m]P	|�  &��A��
* 

Policy/Extrinsic RewardnnfA����&       sO� 	��  &��A��
*

Policy/Entropy�Z�?�Zas5       ��]�	��  &��A��
*&
$
Environment/Cumulative Reward �\A��G�2       $V�	R�  &��A��
*#
!
Environment/Episode Length  �B$��)       7�_ 	8�  &��A��
*

Losses/Value Loss���@���*       ����	P�  &��A��
*

Losses/Policy Loss��<h�ɝ,       ���E	��  &��A��
*

Policy/Learning Rateޤ�9P�67       ���Y	��&��A��
*(
&
Policy/Extrinsic Value Estimate�|U@�W��&       sO� 	��&��A��
*

Policy/Entropy{]�?��Y~5       ��]�	[��&��A��
*&
$
Environment/Cumulative Reward�׋A�3��2       $V�	��&��A��
*#
!
Environment/Episode Length  �B�Š /       m]P	!�&��A��
* 

Policy/Extrinsic Reward�يA
�>?)       7�_ 	O<�&��A��
*

Losses/Value Loss�5�@nW6�*       ����	'J�&��A��
*

Losses/Policy Loss3��<�`]�,       ���E	�Q�&��A��
*

Policy/Learning Rate?z�9~�*7       ���Y	?���'��A��*(
&
Policy/Extrinsic Value Estimate��`@��e/       m]P	����'��A��* 

Policy/Extrinsic Reward*�eAP_�*&       sO� 	���'��A��*

Policy/Entropyuj�?a5       ��]�	���'��A��*&
$
Environment/Cumulative Reward--]AW'�2       $V�	���'��A��*#
!
Environment/Episode Length  �B�M))       7�_ 	���'��A��*

Losses/Value Loss���@�!�*       ����	���'��A��*

Losses/Policy LossY5�<��-,       ���E	B��'��A��*

Policy/Learning RateMP�9��d87       ���Y	8-��(��A��*(
&
Policy/Extrinsic Value Estimate�f@z�K5/       m]P	����(��A��* 

Policy/Extrinsic Reward��`AN�3&       sO� 	���(��A��*

Policy/Entropy�N�?0P:5       ��]�	$��(��A��*&
$
Environment/Cumulative RewardUUgAg��z2       $V�	���(��A��*#
!
Environment/Episode Length  �B�B�)       7�_ 	z��(��A��*

Losses/Value Loss�!�@����*       ����	�0��(��A��*

Losses/Policy Loss6��<(w|,       ���E	v5��(��A��*

Policy/Learning Rate�%�9�vb'7       ���Y	*��$)��A��*(
&
Policy/Extrinsic Value Estimateo�d@1��k&       sO� 	|)�$)��A��*

Policy/Entropy�<�?��4q5       ��]�	�L�$)��A��*&
$
Environment/Cumulative RewardPPPAꞐ�2       $V�	.V�$)��A��*#
!
Environment/Episode Length  �B�2�/       m]P	��$)��A��* 

Policy/Extrinsic Reward�OA\7       ���Y	^N�*��A�*(
&
Policy/Extrinsic Value Estimate�h@Y�p�/       m]P	���*��A�* 

Policy/Extrinsic RewardtT�AF��)       7�_ 	|��*��A�*

Losses/Value Loss�܌@Y&�*       ����	���*��A�*

Losses/Policy Loss�=	�P,       ���E	���*��A�*

Policy/Learning Rate���9L@&       sO� 	c��*��A�*

Policy/Entropy�S�?��5       ��]�	/��*��A�*&
$
Environment/Cumulative Reward���A����2       $V�	���*��A�*#
!
Environment/Episode Length  �B�	҉7       ���Y	G[�+��A��*(
&
Policy/Extrinsic Value Estimate��f@{fvf/       m]P	�n�+��A��* 

Policy/Extrinsic RewarddddA�W�&       sO� 	 u�+��A��*

Policy/Entropy :�?=�)       7�_ 	�z�+��A��*

Losses/Value LossP��@����*       ����	��+��A��*

Losses/Policy Lossֱ�<c��&,       ���E	U��+��A��*

Policy/Learning Rate"љ9���5       ��]�	���+��A��*&
$
Environment/Cumulative RewardU�^A4|>�2       $V�	9��+��A��*#
!
Environment/Episode Length  �B����7       ���Y	[$,��A��*(
&
Policy/Extrinsic Value Estimateĉc@�\)%&       sO� 	��$,��A��*

Policy/Entropy$/�?�ҧ�5       ��]�	(�$,��A��*&
$
Environment/Cumulative RewardssSA�`�e2       $V�	��$,��A��*#
!
Environment/Episode Length  �B���L/       m]P	��$,��A��* 

Policy/Extrinsic Reward�oTA�]�)       7�_ 	��$,��A��*

Losses/Value Loss�|�@=�"r*       ����	��$,��A��*

Losses/Policy Loss�k�<���,       ���E	��$,��A��*

Policy/Learning Rate3��9���=7       ���Y	�ŏb-��A��*(
&
Policy/Extrinsic Value Estimate�;b@�e�]/       m]P	rŐb-��A��* 

Policy/Extrinsic Reward:*`A۝y�&       sO� 	2ːb-��A��*

Policy/Entropy��?@���5       ��]�	�͐b-��A��*&
$
Environment/Cumulative Reward��]A�]�2       $V�	�Аb-��A��*#
!
Environment/Episode Length  �B�-q�)       7�_ 	�Ґb-��A��*

Losses/Value Loss�={@�#�G*       ����	0Ԑb-��A��*

Losses/Policy Loss~ж<Ny��,       ���E	֐b-��A��*

Policy/Learning Rate�|�9�<f�7       ���Y	��9�.��A��*(
&
Policy/Extrinsic Value Estimate��b@]��w/       m]P	�:�.��A��* 

Policy/Extrinsic RewarduAP:X&       sO� 	$B>�.��A��*

Policy/Entropy�ֵ??��5       ��]�	�@B�.��A��*&
$
Environment/Cumulative Reward�*zA�^�2       $V�	IB�.��A��*#
!
Environment/Episode Length  �B-.[7)       7�_ 	��B�.��A��*

Losses/Value Lossrw�@����*       ����	��B�.��A��*

Losses/Policy Loss3:�<ň[�,       ���E	��B�.��A��*

Policy/Learning Rate�R�9�MN�7       ���Y	*~/��A��*(
&
Policy/Extrinsic Value Estimate�n@�Fp�&       sO� 	ܸ~/��A��*

Policy/Entropy���?��+5       ��]�	|�~/��A��*&
$
Environment/Cumulative Reward((�A��zE2       $V�	��~/��A��*#
!
Environment/Episode Length  �B4ȁ/       m]P	��~/��A��* 

Policy/Extrinsic Rewardce�A۩��)       7�_ 	��~/��A��*

Losses/Value Loss4�@*�*       ����	n�~/��A��*

Losses/Policy Loss1�<$��,       ���E	/�~/��A��*

Policy/Learning Rate(�9%���7       ���Y	��&u0��A��*(
&
Policy/Extrinsic Value Estimateh�g@0��5/       m]P	wK'u0��A��* 

Policy/Extrinsic Reward��bAe.(<&       sO� 	 P'u0��A��*

Policy/Entropyስ?#ި�5       ��]�	DS'u0��A��*&
$
Environment/Cumulative Reward--]A)R-�2       $V�	GU'u0��A��*#
!
Environment/Episode Length  �Bw��)       7�_ 	W'u0��A��*

Losses/Value Loss�	�@(�*       ����	�X'u0��A��*

Losses/Policy Loss�j�<^T�,       ���E	Dh'u0��A��*

Policy/Learning Rate��9���7       ���Y	80Wk1��A��*(
&
Policy/Extrinsic Value Estimate��p@1���/       m]P	5�Wk1��A��* 

Policy/Extrinsic RewardPP`A�0��&       sO� 	�Wk1��A��*

Policy/Entropyܛ�?��5       ��]�	D�Wk1��A��*&
$
Environment/Cumulative Reward  fAB�� 2       $V�	ΫWk1��A��*#
!
Environment/Episode Length  �BEA��)       7�_ 	g�Wk1��A��*

Losses/Value Loss���@���*       ����	��Wk1��A��*

Losses/Policy Lossa3�<*g,       ���E	�Wk1��A��*

Policy/Learning RatevӘ9.�17       ���Y	2#�W2��A�*(
&
Policy/Extrinsic Value EstimateC�m@�9�&       sO� 	W2��A�*

Policy/Entropy妵?�4��5       ��]�	ĕ�W2��A�*&
$
Environment/Cumulative RewardiiaAG��2       $V�	���W2��A�*#
!
Environment/Episode Length  �B.��/       m]P	N��W2��A�* 

Policy/Extrinsic RewardQ`AxHZ�)       7�_ 	�W2��A�*

Losses/Value Loss�j�@b"*       ����	���W2��A�*

Losses/Policy Loss{b�<d��",       ���E	���W2��A�*

Policy/Learning Rate���9}Jў7       ���Y	lтR3��A��*(
&
Policy/Extrinsic Value EstimateG�r@�[�/       m]P	�#�R3��A��* 

Policy/Extrinsic Reward��nA���-&       sO� 	�&�R3��A��*

Policy/Entropy�ĵ?�t+5       ��]�	�8�R3��A��*&
$
Environment/Cumulative Reward��oA�/I2       $V�	�<�R3��A��*#
!
Environment/Episode Length  �B��)       7�_ 	�>�R3��A��*

Losses/Value Loss�V�@'�q*       ����	�A�R3��A��*

Losses/Policy Loss���<��,K,       ���E	�D�R3��A��*

Policy/Learning Rate�~�91�_7       ���Y	Ǻ�U4��A��*(
&
Policy/Extrinsic Value Estimate:�@IY�/       m]P	&�U4��A��* 

Policy/Extrinsic Rewardxx�A-�#&       sO� 	5*�U4��A��*

Policy/Entropy�ŵ?��l�5       ��]�	y<�U4��A��*&
$
Environment/Cumulative RewardU�A���o2       $V�	`@�U4��A��*#
!
Environment/Episode Length  �BkjČ)       7�_ 	pC�U4��A��*

Losses/Value LossFT�@1�],*       ����	�V�U4��A��*

Losses/Policy Loss���<�*:�,       ���E	�h�U4��A��*

Policy/Learning Rate�T�9����7       ���Y	z �B5��A��*(
&
Policy/Extrinsic Value Estimate7�@h?;�&       sO� 	|�B5��A��*

Policy/Entropy��?Q�<�5       ��]�	Ä�B5��A��*&
$
Environment/Cumulative Reward�׃A���g2       $V�	���B5��A��*#
!
Environment/Episode Length  �B�^M/       m]P	Ӊ�B5��A��* 

Policy/Extrinsic Rewardz��A�":)       7�_ 	��B5��A��*

Losses/Value Loss��@_�'�*       ����	��B5��A��*

Losses/Policy LossL�<�l3,       ���E	Ǹ�B5��A��*

Policy/Learning Rate[*�9��7       ���Y	^�786��A��*(
&
Policy/Extrinsic Value Estimate@�G�i/       m]P	�!886��A��* 

Policy/Extrinsic Reward�ōA��P&       sO� 	�%886��A��*

Policy/Entropy���?>?\5       ��]�	�5886��A��*&
$
Environment/Cumulative Reward--�A�P4�2       $V�	:9886��A��*#
!
Environment/Episode Length  �B�1Q�)       7�_ 	�I886��A��*

Losses/Value Loss\��@.�V*       ����	4M886��A��*

Losses/Policy Lossᱻ<�3�,       ���E	4O886��A��*

Policy/Learning Ratef �9G6�7       ���Y	��}&7��A��*(
&
Policy/Extrinsic Value Estimatee�~@ϟ��/       m]P	��}&7��A��* 

Policy/Extrinsic Reward��KA���b&       sO� 	K�}&7��A��*

Policy/Entropy�ߵ?�Q�^5       ��]�	h�}&7��A��*&
$
Environment/Cumulative Reward�*PAe�2       $V�	��}&7��A��*#
!
Environment/Episode Length  �BY��)       7�_ 	��}&7��A��*

Losses/Value Loss��@9�X�*       ����	��}&7��A��*

Losses/Policy Loss���<%��,       ���E	��}&7��A��*

Policy/Learning Rate�՗9��g�7       ���Y	lL8��A��*(
&
Policy/Extrinsic Value Estimate��@���&       sO� 	I�L8��A��*

Policy/Entropy�	�?b��5       ��]�	'�L8��A��*&
$
Environment/Cumulative RewardFF^Aȇ2       $V�	�L8��A��*#
!
Environment/Episode Length  �B�}/J/       m]P	��L8��A��* 

Policy/Extrinsic Reward�_A�R])       7�_ 	r�L8��A��*

Losses/Value Loss%��@-�l*       ����	"�L8��A��*

Losses/Policy Lossm��<ţu,       ���E	��L8��A��*

Policy/Learning Rate۫�9=��57       ���Y	z7�9��A��*(
&
Policy/Extrinsic Value Estimater3�@m2H/       m]P	/R�9��A��* 

Policy/Extrinsic Reward��gAd�&       sO� 	�T�9��A��*

Policy/Entropy��?�q~5       ��]�	\V�9��A��*&
$
Environment/Cumulative Reward��gA��]S2       $V�	X�9��A��*#
!
Environment/Episode Length  �B�o)       7�_ 	�Y�9��A��*

Losses/Value Loss@��@*o�*       ����	>[�9��A��*

Losses/Policy Loss8�<�S$�,       ���E	�]�9��A��*

Policy/Learning Rate<��9�<�J7       ���Y	z�R�9��A��*(
&
Policy/Extrinsic Value Estimate�z@�G/       m]P	�%S�9��A��* 

Policy/Extrinsic Reward��cA�W�&       sO� 	 )S�9��A��*

Policy/Entropy#�?[��[5       ��]�	|+S�9��A��*&
$
Environment/Cumulative RewardUUdA��T2       $V�	,.S�9��A��*#
!
Environment/Episode Length  �B�f)       7�_ 	I0S�9��A��*

Losses/Value Loss�*�@k��@*       ����	�5S�9��A��*

Losses/Policy Loss��<G�*,       ���E	e9S�9��A��*

Policy/Learning RateKW�9�ar�7       ���Y	�ȓ�:��A��*(
&
Policy/Extrinsic Value Estimate5�@�<B&       sO� 	sٓ�:��A��*

Policy/Entropy�%�?3 �5       ��]�	�ۓ�:��A��*&
$
Environment/Cumulative Reward��XA���<2       $V�	�ݓ�:��A��*#
!
Environment/Episode Length  �BY}E�/       m]P	~ߓ�:��A��* 

Policy/Extrinsic Reward��YAQ�)       7�_ 	P��:��A��*

Losses/Value Loss�/�@���.*       ����	���:��A��*

Losses/Policy Loss�G�<,H�,       ���E	����:��A��*

Policy/Learning Rate�,�9phYx7       ���Y	3��;��A��*(
&
Policy/Extrinsic Value Estimate�|@ig��/       m]P	{+��;��A��* 

Policy/Extrinsic Reward*�uAɒ/&       sO� 	�;��;��A��*

Policy/Entropy�C�?(��5       ��]�	B?��;��A��*&
$
Environment/Cumulative Reward��uA� ��2       $V�	BA��;��A��*#
!
Environment/Episode Length  �Bۉ��)       7�_ 	C��;��A��*

Losses/Value LossR�@n��1*       ����	�D��;��A��*

Losses/Policy Loss?�<�)h�,       ���E	xF��;��A��*

Policy/Learning Rate��9#x/-7       ���Y	����<��A��*(
&
Policy/Extrinsic Value Estimate��@v�4&/       m]P	j �<��A��* 

Policy/Extrinsic Reward��pA�<V&       sO� 	`ᠷ<��A��*

Policy/Entropy�-�?���|5       ��]�	*堷<��A��*&
$
Environment/Cumulative Reward  rA�Qe2       $V�	%砷<��A��*#
!
Environment/Episode Length  �BC�*)       7�_ 	9頷<��A��*

Losses/Value Loss౴@~���*       ����	�꠷<��A��*

Losses/Policy Lossی�<���,       ���E	x젷<��A��*

Policy/Learning Rateؖ9*� 7       ���Y	����<��A��*(
&
Policy/Extrinsic Value Estimate���@]c�\&       sO� 	���<��A��*

Policy/Entropy�%�?�h�5       ��]�	�	��<��A��*&
$
Environment/Cumulative Reward��pAY$;2       $V�	d:��<��A��*#
!
Environment/Episode Length  �B�i��/       m]P	`>��<��A��* 

Policy/Extrinsic Reward��qA�07       ���Y	�]Z�=��A��*(
&
Policy/Extrinsic Value Estimate�=�@-@k�/       m]P	��Z�=��A��* 

Policy/Extrinsic Reward��YA�ǜ�)       7�_ 	
�Z�=��A��*

Losses/Value Loss��@���*       ����	7�Z�=��A��*

Losses/Policy Loss3h�<���,       ���E	 �Z�=��A��*

Policy/Learning Rate-��9�׃�&       sO� 	��Z�=��A��*

Policy/Entropyn%�?�C�)5       ��]�	y�Z�=��A��*&
$
Environment/Cumulative Reward��VA�& 2       $V�	q�Z�=��A��*#
!
Environment/Episode Length  �Bs#HW7       ���Y	:C��>��A��*(
&
Policy/Extrinsic Value Estimate T�@]l��/       m]P	����>��A��* 

Policy/Extrinsic Reward

zA���&       sO� 	����>��A��*

Policy/Entropy�I�?���)       7�_ 	���>��A��*

Losses/Value Loss�@�~*       ����	����>��A��*

Losses/Policy LossA��<�j,       ���E	���>��A��*

Policy/Learning Rate���9:��5       ��]�	a���>��A��*&
$
Environment/Cumulative Reward�*�A�4�^2       $V�	����>��A��*#
!
Environment/Episode Length  �B(4k7       ���Y	EH+�?��Aл*(
&
Policy/Extrinsic Value Estimatet��@j�N>&       sO� 	}�+�?��Aл*

Policy/Entropy�S�?a2-5       ��]�	�+�?��Aл*&
$
Environment/Cumulative Reward  XA<Qsl2       $V�	��+�?��Aл*#
!
Environment/Episode Length  �Bt��/       m]P	M�+�?��Aл* 

Policy/Extrinsic Reward�~VA�ڗ)       7�_ 	<�+�?��Aл*

Losses/Value Loss!1A�r*       ����	��+�?��Aл*

Losses/Policy Loss�H�<]� �,       ���E	�+�?��Aл*

Policy/Learning Rate�Y�9�SG�7       ���Y	�m��@��A��*(
&
Policy/Extrinsic Value Estimate�1�@�t�/       m]P	����@��A��* 

Policy/Extrinsic Reward  pAx;�&       sO� 	����@��A��*

Policy/Entropy�&�?:�5       ��]�	����@��A��*&
$
Environment/Cumulative RewardsskA&!y2       $V�	K��@��A��*#
!
Environment/Episode Length  �Bi���)       7�_ 	���@��A��*

Losses/Value Loss���@,�J*       ����	x��@��A��*

Losses/Policy LossT(�<��,       ���E	<0��@��A��*

Policy/Learning Rate/�9���i7       ���Y	�lԩA��A��*(
&
Policy/Extrinsic Value Estimate@9D�v/       m]P	թA��A��* 

Policy/Extrinsic Reward

rAܞ�&       sO� 	YթA��A��*

Policy/Entropy0@�?���25       ��]�	�)թA��A��*&
$
Environment/Cumulative Reward��uA�?͞2       $V�	<թA��A��*#
!
Environment/Episode Length  �Bs��)       7�_ 	�?թA��A��*

Losses/Value Loss�&�@Rx|#*       ����	5BթA��A��*

Losses/Policy LossD^�<(�rr,       ���E	gDթA��A��*

Policy/Learning Rate�9T4&�7       ���Y	@���B��A��*(
&
Policy/Extrinsic Value Estimate8�@XC��&       sO� 	�N��B��A��*

Policy/EntropyA\�?I��5       ��]�	�\��B��A��*&
$
Environment/Cumulative RewardiiAA.;�2       $V�	t`��B��A��*#
!
Environment/Episode Length  �BΑ_v/       m]P	�b��B��A��* 

Policy/Extrinsic Reward�7BAt ��)       7�_ 	�d��B��A��*

Losses/Value Loss���@W0;�*       ����	g��B��A��*

Losses/Policy Loss0Š<B5'�,       ���E	/m��B��A��*

Policy/Learning Ratesڕ9���7       ���Y	!�C��A��*(
&
Policy/Extrinsic Value Estimate���@�;j�/       m]P	��C��A��* 

Policy/Extrinsic Reward��gA��I�&       sO� 	!��C��A��*

Policy/Entropy�d�?�=S5       ��]�	1��C��A��*&
$
Environment/Cumulative Reward��cA�Ä�2       $V�	��C��A��*#
!
Environment/Episode Length  �B~Y��)       7�_ 	���C��A��*

Losses/Value Loss1��@?ۮ�*       ����	��C��A��*

Losses/Policy LossY��<����,       ���E	*��C��A��*

Policy/Learning Rate���9�Y��7       ���Y	 �;zD��A��*(
&
Policy/Extrinsic Value Estimate��}@<l�/       m]P	�[<zD��A��* 

Policy/Extrinsic Reward��WA ��&       sO� 	�^<zD��A��*

Policy/EntropyA�?	ME�5       ��]�	�`<zD��A��*&
$
Environment/Cumulative Reward �\A��O�2       $V�	.s<zD��A��*#
!
Environment/Episode Length  �B)x�)       7�_ 	Wy<zD��A��*

Losses/Value Loss#%�@@�Ϝ*       ����	|<zD��A��*

Losses/Policy Loss�s�<�m��,       ���E	S�<zD��A��*

Policy/Learning Rate⅕9�a�?7       ���Y	K�zkE��A��*(
&
Policy/Extrinsic Value Estimate��a@�2)&       sO� 	�R{kE��A��*

Policy/Entropy 6�?u��5       ��]�	�|{kE��A��*&
$
Environment/Cumulative Reward��*A�0�2       $V�	�{kE��A��*#
!
Environment/Episode Length  �BK��/       m]P	�{kE��A��* 

Policy/Extrinsic RewardB�*A%w��)       7�_ 	s�{kE��A��*

Losses/Value Loss{�@T��U*       ����	��{kE��A��*

Losses/Policy Loss3I�<����,       ���E	{kE��A��*

Policy/Learning Rate�[�9Vcޯ7       ���Y	�5�ZF��A��*(
&
Policy/Extrinsic Value Estimate�8�@�"�/       m]P	�k�ZF��A��* 

Policy/Extrinsic Reward�hAߑ�&       sO� 	ao�ZF��A��*

Policy/Entropy�,�?P��<5       ��]�	zq�ZF��A��*&
$
Environment/Cumulative Reward��hAJ�y&2       $V�	_��ZF��A��*#
!
Environment/Episode Length  �B[�)       7�_ 	���ZF��A��*

Losses/Value Lossܚ�@�x�X*       ����	솙ZF��A��*

Losses/Policy Loss���<N|��,       ���E	���ZF��A��*

Policy/Learning RateU1�9Y��7       ���Y	��JG��AЬ *(
&
Policy/Extrinsic Value EstimateJ~k@)��/       m]P	�-�JG��AЬ * 

Policy/Extrinsic Reward##cAC6F&       sO� 	1�JG��AЬ *

Policy/Entropy
�?=7$^5       ��]�	4�JG��AЬ *&
$
Environment/Cumulative RewardU�cA�<O�2       $V�	�B�JG��AЬ *#
!
Environment/Episode Length  �B���@)       7�_ 	1F�JG��AЬ *

Losses/Value LossGM�@���K*       ����	kH�JG��AЬ *

Losses/Policy Loss��<��,       ���E	�J�JG��AЬ *

Policy/Learning Ratec�9��S7       ���Y	S;L:H��A�� *(
&
Policy/Extrinsic Value Estimatex�x@�q�&       sO� 	qL:H��A�� *

Policy/Entropy�ܵ?��hr5       ��]�	$�L:H��A�� *&
$
Environment/Cumulative Reward((XA����2       $V�	�L:H��A�� *#
!
Environment/Episode Length  �BM9;/       m]P	�L:H��A�� * 

Policy/Extrinsic RewardL:UA�}�)       7�_ 	�L:H��A�� *

Losses/Value Loss��@���*       ����	ΉL:H��A�� *

Losses/Policy LossLv�<�*i�,       ���E	u�L:H��A�� *

Policy/Learning Rate�ܔ9�A~J7       ���Y	���&I��A��!*(
&
Policy/Extrinsic Value Estimate��n@�]+s/       m]P	a��&I��A��!* 

Policy/Extrinsic Reward��UA0-xg&       sO� 	���&I��A��!*

Policy/Entropy��?�.��5       ��]�	`��&I��A��!*&
$
Environment/Cumulative RewardAAYA��Y]2       $V�	%��&I��A��!*#
!
Environment/Episode Length  �B�F�3)       7�_ 	E��&I��A��!*

Losses/Value Lossfݺ@\�s&*       ����	+��&I��A��!*

Losses/Policy Loss�<e�},       ���E	�ߗ&I��A��!*

Policy/Learning Rateֲ�9�cњ7       ���Y	�;�J��A��"*(
&
Policy/Extrinsic Value Estimate�Nm@�44�/       m]P	�M�J��A��"* 

Policy/Extrinsic Reward��TAo���&       sO� 	�O�J��A��"*

Policy/Entropy��?�6��5       ��]�	�Q�J��A��"*&
$
Environment/Cumulative Reward��MA@s�@2       $V�	�S�J��A��"*#
!
Environment/Episode Length  �BnM��)       7�_ 	TU�J��A��"*

Losses/Value Loss���@�.n3*       ����	�V�J��A��"*

Losses/Policy Loss:A=��,       ���E	�m�J��A��"*

Policy/Learning Rate:��9�۸7       ���Y	sdK��A��"*(
&
Policy/Extrinsic Value Estimateqeo@&LM�&       sO� 	o�K��A��"*

Policy/EntropyWo�?���5       ��]�	��K��A��"*&
$
Environment/Cumulative Reward��dAH��?2       $V�	��K��A��"*#
!
Environment/Episode Length  �B����/       m]P	ҎK��A��"* 

Policy/Extrinsic Reward�beAI�>r)       7�_ 	��K��A��"*

Losses/Value Loss���@7P�*       ����	h�K��A��"*

Losses/Policy Lossk�<-{��,       ���E	ĕK��A��"*

Policy/Learning RateE^�9��\7       ���Y	��9�K��A��#*(
&
Policy/Extrinsic Value Estimate�U`@��/       m]P	�p:�K��A��#* 

Policy/Extrinsic RewardӰMA�ܷ�&       sO� 	�s:�K��A��#*

Policy/Entropydz�?u��G5       ��]�	Sv:�K��A��#*&
$
Environment/Cumulative RewardssKA��m2       $V�	�y:�K��A��#*#
!
Environment/Episode Length  �B`�)       7�_ 	%|:�K��A��#*

Losses/Value Loss�U�@{_)*       ����	�~:�K��A��#*

Losses/Policy Loss�#�<gP[�,       ���E	��:�K��A��#*

Policy/Learning Rate�3�9�<��7       ���Y	��.�L��A��$*(
&
Policy/Extrinsic Value Estimate�}s@u�D�/       m]P	\�.�L��A��$* 

Policy/Extrinsic RewardPP`A��ڼ&       sO� 	`/�L��A��$*

Policy/EntropyTi�?�I�_5       ��]�	\/�L��A��$*&
$
Environment/Cumulative Reward �dA�,C�2       $V�	/�L��A��$*#
!
Environment/Episode Length  �B�>4%)       7�_ 	�/�L��A��$*

Losses/Value Loss'u�@h��*       ����	t
/�L��A��$*

Losses/Policy Loss2�<�Ջ�,       ���E	/�L��A��$*

Policy/Learning Rate�	�96 n�7       ���Y	
ֱ�M��A��$*(
&
Policy/Extrinsic Value Estimate���@x07^&       sO� 	���M��A��$*

Policy/Entropy'R�?�)�5       ��]�	����M��A��$*&
$
Environment/Cumulative Reward___A�r2       $V�	p��M��A��$*#
!
Environment/Episode Length  �B�X�f/       m]P	O#��M��A��$* 

Policy/Extrinsic Reward!w]AZ9~�)       7�_ 	i8��M��A��$*

Losses/Value Loss���@F�B=*       ����	�<��M��A��$*

Losses/Policy Loss=��<Gp�,       ���E	L��M��A��$*

Policy/Learning Rateߓ9D�#7       ���Y	����N��AН%*(
&
Policy/Extrinsic Value Estimate��@4��/       m]P	}���N��AН%* 

Policy/Extrinsic Reward�D�A#���&       sO� 	����N��AН%*

Policy/Entropy}Y�?�x@�5       ��]�	^���N��AН%*&
$
Environment/Cumulative Rewardss�ArJ�B2       $V�	#���N��AН%*#
!
Environment/Episode Length  �B��AC)       7�_ 	����N��AН%*

Losses/Value Loss9`AT )*       ����	����N��AН%*

Losses/Policy Loss��<���,       ���E	j���N��AН%*

Policy/Learning Rate*��9'4w�7       ���Y	:q�O��A��%*(
&
Policy/Extrinsic Value Estimate��@0Bҥ/       m]P	�kq�O��A��%* 

Policy/Extrinsic RewardKKsAX*&       sO� 	oq�O��A��%*

Policy/Entropy�C�?8��"5       ��]�	�qq�O��A��%*&
$
Environment/Cumulative Reward�*vA���2       $V�	�tq�O��A��%*#
!
Environment/Episode Length  �B�;)       7�_ 	�yq�O��A��%*

Losses/Value Loss?A�@�Y٪*       ����	�|q�O��A��%*

Losses/Policy Loss�A�<���,       ���E	�~q�O��A��%*

Policy/Learning Rate���9�T�7       ���Y	!��O��A�&*(
&
Policy/Extrinsic Value Estimate�b�@��
[&       sO� 	#��O��A�&*

Policy/Entropy��?bC$�5       ��]�	�%��O��A�&*&
$
Environment/Cumulative Reward_A��g�2       $V�	f,��O��A�&*#
!
Environment/Episode Length  �B���/       m]P	�.��O��A�&* 

Policy/Extrinsic Reward�(`AA<�f7       ���Y	���P��A��'*(
&
Policy/Extrinsic Value Estimate�Ay@!ګ�/       m]P	Q.�P��A��'* 

Policy/Extrinsic Rewardq�SA���)       7�_ 	1�P��A��'*

Losses/Value Loss�@�@c�w*       ����	3�P��A��'*

Losses/Policy Losss׺<h�V�,       ���E	A�P��A��'*

Policy/Learning Rate�`�9�s�-&       sO� 	�C�P��A��'*

Policy/Entropy��?��lU5       ��]�	�E�P��A��'*&
$
Environment/Cumulative Reward��TAK��#2       $V�	�G�P��A��'*#
!
Environment/Episode Length  �B
̞7       ���Y	Ab��Q��A��'*(
&
Policy/Extrinsic Value Estimate�e�@/�5/       m]P	���Q��A��'* 

Policy/Extrinsic Reward<<dA�$��&       sO� 	W��Q��A��'*

Policy/Entropy��?����)       7�_ 	���Q��A��'*

Losses/Value LossJ+�@�#xV*       ����	iS��Q��A��'*

Losses/Policy Loss̀�<Ը0l,       ���E	�W��Q��A��'*

Policy/Learning Rate�5�9�P��5       ��]�	�{��Q��A��'*&
$
Environment/Cumulative Reward��eA1�ُ2       $V�	���Q��A��'*#
!
Environment/Episode Length  �B��d37       ���Y	g\��R��A��(*(
&
Policy/Extrinsic Value Estimate�ϐ@-�Ν&       sO� 	Ք��R��A��(*

Policy/Entropy��?�85       ��]�	����R��A��(*&
$
Environment/Cumulative Rewardxx�Aa�b�2       $V�	����R��A��(*#
!
Environment/Episode Length  �B���	/       m]P	����R��A��(* 

Policy/Extrinsic Reward���A?�)       7�_ 	z���R��A��(*

Losses/Value Loss��@���**       ����	)���R��A��(*

Losses/Policy Lossy(�<u��,       ���E	���R��A��(*

Policy/Learning Rate�9��q�7       ���Y	T�f�S��A��(*(
&
Policy/Extrinsic Value Estimate��@����/       m]P	K�f�S��A��(* 

Policy/Extrinsic Reward�LsA跾H&       sO� 	d�f�S��A��(*

Policy/Entropy��?�F�5       ��]�	k�f�S��A��(*&
$
Environment/Cumulative RewardmA��v�2       $V�	(�f�S��A��(*#
!
Environment/Episode Length  �B�y{)       7�_ 	'�f�S��A��(*

Losses/Value Loss{�A-K�o*       ����	 g�S��A��(*

Losses/Policy Losso�~<���,       ���E	�g�S��A��(*

Policy/Learning Ratem�9���Z7       ���Y	&�]�T��A��)*(
&
Policy/Extrinsic Value Estimate�ҍ@�}/       m]P	fK^�T��A��)* 

Policy/Extrinsic Reward�A�[�&       sO� 	DN^�T��A��)*

Policy/Entropy��?�/:5       ��]�	P^�T��A��)*&
$
Environment/Cumulative Reward�j�A�a\�2       $V�	�Q^�T��A��)*#
!
Environment/Episode Length  �B^��)       7�_ 	vS^�T��A��)*

Losses/Value Loss�A��a3*       ����	�b^�T��A��)*

Losses/Policy Loss'N�<���X,       ���E	�e^�T��A��)*

Policy/Learning Rate~��9:.M�7       ���Y	�r�U��AЎ**(
&
Policy/Extrinsic Value Estimate��@�(��&       sO� 	�r�U��AЎ**

Policy/Entropy�?�yW�5       ��]�	��r�U��AЎ**&
$
Environment/Cumulative Reward��yA�B��2       $V�	�s�U��AЎ**#
!
Environment/Episode Length  �B@���/       m]P	�,s�U��AЎ** 

Policy/Extrinsic Reward��yA���)       7�_ 	�0s�U��AЎ**

Losses/Value Loss+�A')*       ����	�2s�U��AЎ**

Losses/Policy Loss���<)f�,       ���E	�s�U��AЎ**

Policy/Learning Rate���9���7       ���Y	�GͤV��A��**(
&
Policy/Extrinsic Value Estimate��@d쉛/       m]P	l�ͤV��A��** 

Policy/Extrinsic Reward�
�A�~y9&       sO� 	c�ͤV��A��**

Policy/Entropy(�?����5       ��]�	A�ͤV��A��**&
$
Environment/Cumulative Reward���A��R2       $V�	�ͤV��A��**#
!
Environment/Episode Length  �B�_84)       7�_ 	��ͤV��A��**

Losses/Value Loss���@踯*       ����	Y�ͤV��A��**

Losses/Policy Loss�K�<^�"r,       ���E	פͤV��A��**

Policy/Learning Rate�b�9�/�7       ���Y	z�W��A�+*(
&
Policy/Extrinsic Value Estimate.^�@k�D/       m]P	�_�W��A�+* 

Policy/Extrinsic Reward��yAT��b&       sO� 	�l�W��A�+*

Policy/Entropy!C�?���5       ��]�	k|�W��A�+*&
$
Environment/Cumulative Reward  zAiZ��2       $V�	E��W��A�+*#
!
Environment/Episode Length  �B�S~e)       7�_ 	Ҏ�W��A�+*

Losses/Value Loss�.Aʯ��*       ����	��W��A�+*

Losses/Policy Loss"C�<tJ,       ���E	��W��A�+*

Policy/Learning RateR8�9گ<7       ���Y	͔��X��A��+*(
&
Policy/Extrinsic Value EstimatedΞ@�J��&       sO� 	[靎X��A��+*

Policy/Entropy�7�?�]�5       ��]�	�읎X��A��+*&
$
Environment/Cumulative Reward���A���2       $V�	9��X��A��+*#
!
Environment/Episode Length  �Bs���/       m]P	fJ��X��A��+* 

Policy/Extrinsic RewardY�A��)       7�_ 	IN��X��A��+*

Losses/Value Loss�A�d��*       ����	jP��X��A��+*

Losses/Policy Loss ��<�҂�,       ���E	!]��X��A��+*

Policy/Learning Ratea�9D-67       ���Y	�IۂY��A��,*(
&
Policy/Extrinsic Value Estimate�ȕ@z� �/       m]P	S�ۂY��A��,* 

Policy/Extrinsic Reward�
�A*৸&       sO� 	��ۂY��A��,*

Policy/Entropy��?d�5       ��]�	g�ۂY��A��,*&
$
Environment/Cumulative Reward22�A��q2       $V�	��ۂY��A��,*#
!
Environment/Episode Length  �B���)       7�_ 	T܂Y��A��,*

Losses/Value Loss��$A��6�*       ����	G܂Y��A��,*

Losses/Policy LossL�<y��,       ���E	�܂Y��A��,*

Policy/Learning Rate��9��W7       ���Y	��wZ��A��-*(
&
Policy/Extrinsic Value Estimate�ϝ@_<��/       m]P	�q�wZ��A��-* 

Policy/Extrinsic Reward�ׇAl���&       sO� 	�w�wZ��A��-*

Policy/Entropy���?YE��5       ��]�	���wZ��A��-*&
$
Environment/Cumulative Reward�*�AH>�2       $V�	���wZ��A��-*#
!
Environment/Episode Length  �B@)       7�_ 	��wZ��A��-*

Losses/Value Loss,%A�N�*       ����	a�wZ��A��-*

Losses/Policy LossN`�<����,       ���E	�+�wZ��A��-*

Policy/Learning Rateҹ�9��}�7       ���Y	���g[��A��-*(
&
Policy/Extrinsic Value Estimate�͟@ܫ�&       sO� 	\�g[��A��-*

Policy/Entropy�ɴ?�c� 5       ��]�	s��g[��A��-*&
$
Environment/Cumulative Reward22�A̛?2       $V�	��g[��A��-*#
!
Environment/Episode Length  �B���o/       m]P	i�g[��A��-* 

Policy/Extrinsic Reward�A��;b)       7�_ 	��g[��A��-*

Losses/Value Loss��A�d��*       ����	�
�g[��A��-*

Losses/Policy Loss,��<`qB,       ���E	�g[��A��-*

Policy/Learning Rate4��9Ct$7       ���Y	{�Y\��A��.*(
&
Policy/Extrinsic Value Estimate�@,k|�/       m]P	�W�Y\��A��.* 

Policy/Extrinsic Reward��A��r�&       sO� 	�Z�Y\��A��.*

Policy/Entropy4��?�D�^5       ��]�	�\�Y\��A��.*&
$
Environment/Cumulative Reward77�A�o��2       $V�	�^�Y\��A��.*#
!
Environment/Episode Length  �BhF)       7�_ 	�`�Y\��A��.*

Losses/Value Loss�)A.�F�*       ����	Ym�Y\��A��.*

Losses/Policy Loss��<�-��,       ���E	&q�Y\��A��.*

Policy/Learning RateBe�9��r�7       ���Y	-t;J]��A��.*(
&
Policy/Extrinsic Value Estimate�l�@⋔�/       m]P	�;J]��A��.* 

Policy/Extrinsic RewardKK�AK,ͅ&       sO� 	q<J]��A��.*

Policy/Entropyѱ�?X�5       ��]�	K!<J]��A��.*&
$
Environment/Cumulative Reward ��Ar�=2       $V�	`#<J]��A��.*#
!
Environment/Episode Length  �Bɣu�)       7�_ 	x&<J]��A��.*

Losses/Value Loss{�!A�9Q*       ����	�(<J]��A��.*

Losses/Policy Loss���<�"Ǆ,       ���E	�*<J]��A��.*

Policy/Learning Rate�:�9���7       ���Y	�9^��A��/*(
&
Policy/Extrinsic Value EstimateFݰ@���&       sO� 	h#9^��A��/*

Policy/Entropy�Ŵ?"P�5       ��]�	�&9^��A��/*&
$
Environment/Cumulative Reward��A�jj�2       $V�	�,9^��A��/*#
!
Environment/Episode Length  �B�ן�/       m]P	�>9^��A��/* 

Policy/Extrinsic Reward�ϞA�-��)       7�_ 	B9^��A��/*

Losses/Value Loss`
LA�r+c*       ����	�D9^��A��/*

Losses/Policy Loss�˲<��$:,       ���E	G9^��A��/*

Policy/Learning Rate��9���7       ���Y	��)-_��A�0*(
&
Policy/Extrinsic Value Estimateﲪ@
��/       m]P	f�)-_��A�0* 

Policy/Extrinsic Reward�L�A[K7w&       sO� 	�)-_��A�0*

Policy/EntropyW��?��C�5       ��]�	��)-_��A�0*&
$
Environment/Cumulative Reward

�Aq�mX2       $V�	��)-_��A�0*#
!
Environment/Episode Length  �B��)       7�_ 	�)-_��A�0*

Losses/Value Loss��3A���*       ����	N�)-_��A�0*

Losses/Policy Loss@�<�,Z.,       ���E	4�)-_��A�0*

Policy/Learning Rate�9$R��7       ���Y	��x!`��A��0*(
&
Policy/Extrinsic Value Estimatet٩@,hI/       m]P	��x!`��A��0* 

Policy/Extrinsic RewardAA�A��&       sO� 	��x!`��A��0*

Policy/Entropyjs�?�ץ�5       ��]�	��x!`��A��0*&
$
Environment/Cumulative Reward  �A<sVl2       $V�	��x!`��A��0*#
!
Environment/Episode Length  �By�)       7�_ 	��x!`��A��0*

Losses/Value Loss�q	A⅌*       ����	U�x!`��A��0*

Losses/Policy Loss��<U�w�,       ���E	��x!`��A��0*

Policy/Learning Rate%��9y�g7       ���Y	��a��A��1*(
&
Policy/Extrinsic Value Estimate\��@��&       sO� 	�ܭa��A��1*

Policy/EntropyP�?�{,�5       ��]�	8�a��A��1*&
$
Environment/Cumulative Reward�ܜA`�f�2       $V�	��a��A��1*#
!
Environment/Episode Length  �B�}*/       m]P	z��a��A��1* 

Policy/Extrinsic Reward3|�A�Ϣ)       7�_ 	�a��A��1*

Losses/Value Loss�A�*�*       ����	[�a��A��1*

Losses/Policy Loss܃�<+V��,       ���E	S	�a��A��1*

Policy/Learning Rate���9���|7       ���Y	��"	b��A��2*(
&
Policy/Extrinsic Value EstimateH��@��</       m]P	��"	b��A��2* 

Policy/Extrinsic Rewardgy�A@�{&       sO� 	��"	b��A��2*

Policy/Entropy��?k`U5       ��]�	��"	b��A��2*&
$
Environment/Cumulative Reward���Aƭ)%2       $V�	U�"	b��A��2*#
!
Environment/Episode Length  �B��U)       7�_ 	��"	b��A��2*

Losses/Value Loss*k.A��*       ����	��"	b��A��2*

Losses/Policy Loss`[�<~~�,       ���E	P�"	b��A��2*

Policy/Learning Rate�g�9�J��7       ���Y	���
c��A��2*(
&
Policy/Extrinsic Value EstimateO��@!�(/       m]P	��
c��A��2* 

Policy/Extrinsic Rewardii�AzEi�&       sO� 	f��
c��A��2*

Policy/Entropy]��?!�*Z5       ��]�	���
c��A��2*&
$
Environment/Cumulative Reward���A@�T�2       $V�	��
c��A��2*#
!
Environment/Episode Length  �Bj���)       7�_ 	l��
c��A��2*

Losses/Value Loss��>A�4�*       ����	��
c��A��2*

Losses/Policy LossGb�<�ȑ,       ���E	F��
c��A��2*

Policy/Learning Rate�<�9�Ĳ�7       ���Y	J��Gc��A��3*(
&
Policy/Extrinsic Value EstimateC$�@
\V&       sO� 	���Gc��A��3*

Policy/Entropy�X�?nh�V5       ��]�	ϝ�Gc��A��3*&
$
Environment/Cumulative Reward22�A��[W2       $V�	蟧Gc��A��3*#
!
Environment/Episode Length  �Bh�?/       m]P	���Gc��A��3* 

Policy/Extrinsic Reward߈�A,��K7       ���Y	�I(>d��A��3*(
&
Policy/Extrinsic Value Estimate���@�_��/       m]P	܄(>d��A��3* 

Policy/Extrinsic Reward���A��-`)       7�_ 	A�(>d��A��3*

Losses/Value Loss��jA|Ē'*       ����	]�(>d��A��3*

Losses/Policy Loss-i�<//[,       ���E	��(>d��A��3*

Policy/Learning Rate
�9\�O&       sO� 	��(>d��A��3*

Policy/Entropy17�?��m5       ��]�	u�(>d��A��3*&
$
Environment/Cumulative Reward__�Aπ��2       $V�	?�(>d��A��3*#
!
Environment/Episode Length  �B��ΐ7       ���Y	�}00e��A�4*(
&
Policy/Extrinsic Value Estimate!��@�n/       m]P	G=10e��A�4* 

Policy/Extrinsic Reward���A���^&       sO� 	}@10e��A�4*

Policy/Entropy.ղ?�ߛ�)       7�_ 	�C10e��A�4*

Losses/Value Lossk5]AW�4�*       ����	G10e��A�4*

Losses/Policy Loss�<R��,       ���E	�T10e��A�4*

Policy/Learning Ratek�9;
_�5       ��]�	GX10e��A�4*&
$
Environment/Cumulative Reward  �AR2+N2       $V�	�h10e��A�4*#
!
Environment/Episode Length  �Bm�i�7       ���Y	K�"f��A��5*(
&
Policy/Extrinsic Value Estimate�.�@?F��&       sO� 	Á�"f��A��5*

Policy/Entropy䶲?��a5       ��]�	
��"f��A��5*&
$
Environment/Cumulative Rewarddd�A�e�;2       $V�	��"f��A��5*#
!
Environment/Episode Length  �B9��/       m]P	��"f��A��5* 

Policy/Extrinsic Reward�<�AY`ǃ)       7�_ 	 ��"f��A��5*

Losses/Value LossJ�OA��=�*       ����	Ő�"f��A��5*

Losses/Policy Lossc��<�ˀ<,       ���E	�ќ"f��A��5*

Policy/Learning Rate{��9�n�7       ���Y	e�Cg��A��5*(
&
Policy/Extrinsic Value Estimate���@ȋ��/       m]P	�Dg��A��5* 

Policy/Extrinsic Reward���A��o�&       sO� 	�	Dg��A��5*

Policy/Entropyu�?�1ʆ5       ��]�	cDg��A��5*&
$
Environment/Cumulative Reward���Ayn��2       $V�	Dg��A��5*#
!
Environment/Episode Length  �B�8:�)       7�_ 	Dg��A��5*

Losses/Value Loss�kA� Q*       ����	�Dg��A��5*

Losses/Policy Loss���<L��X,       ���E	wDg��A��5*

Policy/Learning Rateݓ�9H8L7       ���Y	��.h��A��6*(
&
Policy/Extrinsic Value Estimate�q�@C��/       m]P	X�.h��A��6* 

Policy/Extrinsic RewardFF�An��j&       sO� 	1�.h��A��6*

Policy/Entropya�?�	]�5       ��]�	t�.h��A��6*&
$
Environment/Cumulative Reward�j�A��,�2       $V�	�.h��A��6*#
!
Environment/Episode Length  �B:h��)       7�_ 	��.h��A��6*

Losses/Value Loss@izAq܍O*       ����	�/h��A��6*

Losses/Policy Loss.�<�[2�,       ���E	/h��A��6*

Policy/Learning Rate�i�9��0�7       ���Y	�#|�h��A��6*(
&
Policy/Extrinsic Value Estimate ��@R;[f&       sO� 	�}|�h��A��6*

Policy/Entropyaӱ?gUv5       ��]�	�|�h��A��6*&
$
Environment/Cumulative Reward�A|-	C2       $V�	��|�h��A��6*#
!
Environment/Episode Length  �B^��]/       m]P	Ӆ|�h��A��6* 

Policy/Extrinsic Reward�b�A�	+)       7�_ 	��|�h��A��6*

Losses/Value Loss'�cA�5��*       ����	X�|�h��A��6*

Losses/Policy Loss�z�<��"�,       ���E	W�|�h��A��6*

Policy/Learning RateM?�9|�7       ���Y	���j��A��7*(
&
Policy/Extrinsic Value EstimatePz�@1ҡL/       m]P	�l�j��A��7* 

Policy/Extrinsic RewardJd�AwZ�R&       sO� 	�p�j��A��7*

Policy/Entropy��?`PT�5       ��]�	�r�j��A��7*&
$
Environment/Cumulative Reward�AI���2       $V�	�t�j��A��7*#
!
Environment/Episode Length  �BZ�<�)       7�_ 	�v�j��A��7*

Losses/Value Loss���A�~��*       ����	���j��A��7*

Losses/Policy Loss0m�<���],       ���E	4��j��A��7*

Policy/Learning Rate^�90�A�7       ���Y	��k��A��8*(
&
Policy/Extrinsic Value EstimateR"�@"�q</       m]P	���k��A��8* 

Policy/Extrinsic Reward���A����&       sO� 	���k��A��8*

Policy/Entropy�ұ?�>�5       ��]�	y��k��A��8*&
$
Environment/Cumulative Reward ��A/�#S2       $V�	���k��A��8*#
!
Environment/Episode Length  �B���)       7�_ 	4��k��A��8*

Losses/Value Lossm*UA,�<R*       ����	��k��A��8*

Losses/Policy LossV��<�@jt,       ���E	���k��A��8*

Policy/Learning Rate��9�m�}7       ���Y	�dl��A��8*(
&
Policy/Extrinsic Value Estimate\��@��W�&       sO� 	1Eel��A��8*

Policy/Entropy�ȱ?�'�x5       ��]�	EJel��A��8*&
$
Environment/Cumulative Reward__B�*U2       $V�	]Nel��A��8*#
!
Environment/Episode Length  �B�T!C/       m]P	�Qel��A��8* 

Policy/Extrinsic RewardJ�B`�)       7�_ 	�Wel��A��8*

Losses/Value LossDcyA�*       ����	FZel��A��8*

Losses/Policy Loss���<��Z,       ���E	�\el��A��8*

Policy/Learning Rate���9\ӟ7       ���Y	��Dm��A�9*(
&
Policy/Extrinsic Value Estimate}��@<>�:/       m]P	�TEm��A�9* 

Policy/Extrinsic RewardR��AQܻ~&       sO� 	�jEm��A�9*

Policy/Entropyꇱ?�G35       ��]�	epEm��A�9*&
$
Environment/Cumulative Reward���A��2       $V�	�uEm��A�9*#
!
Environment/Episode Length  �B�~ �)       7�_ 	�zEm��A�9*

Losses/Value Loss��A���*       ����	9}Em��A�9*

Losses/Policy Losst1�<�j�,       ���E	ŌEm��A�9*

Policy/Learning Rate1��9#/�n7       ���Y	ц�m��A��9*(
&
Policy/Extrinsic Value Estimate�;�@y�/       m]P	%��m��A��9* 

Policy/Extrinsic Rewarddd B�j��&       sO� 	���m��A��9*

Policy/Entropy�D�?j��*5       ��]�	����m��A��9*&
$
Environment/Cumulative Reward���AQwl,2       $V�	0��m��A��9*#
!
Environment/Episode Length  �B���)       7�_ 	��m��A��9*

Losses/Value Loss���A�*s*       ����	��m��A��9*

Losses/Policy LossZ��<K^��,       ���E	���m��A��9*

Policy/Learning Rate@l�9�*�7       ���Y	� �n��A��:*(
&
Policy/Extrinsic Value Estimate�p�@N�R&       sO� 	�w �n��A��:*

Policy/Entropy��?Vu@(5       ��]�	�{ �n��A��:*&
$
Environment/Cumulative Reward��BXD`y2       $V�	V �n��A��:*#
!
Environment/Episode Length  �Bm4]/       m]P	l� �n��A��:* 

Policy/Extrinsic Reward[� B��)       7�_ 	G� �n��A��:*

Losses/Value Loss���A�Q*       ����	�� �n��A��:*

Losses/Policy LossA��<y�:�,       ���E	�� �n��A��:*

Policy/Learning Rate�A�9���7       ���Y	�^�o��A��;*(
&
Policy/Extrinsic Value Estimate���@��r�/       m]P	v��o��A��;* 

Policy/Extrinsic Reward��A�N&       sO� 	$��o��A��;*

Policy/Entropy$��?�
P5       ��]�	���o��A��;*&
$
Environment/Cumulative Reward(( B���s2       $V�	D��o��A��;*#
!
Environment/Episode Length  �B}ڃ�)       7�_ 	���o��A��;*

Losses/Value Loss��A<�� *       ����	���o��A��;*

Losses/Policy LossE��<j�O,       ���E	���o��A��;*

Policy/Learning Rate��9^�N�7       ���Y	J}�p��A��;*(
&
Policy/Extrinsic Value Estimate'_�@��wb/       m]P	�u}�p��A��;* 

Policy/Extrinsic Reward��B���&       sO� 	�x}�p��A��;*

Policy/Entropy#h�?��5�5       ��]�	�z}�p��A��;*&
$
Environment/Cumulative Reward �Bh���2       $V�	:|}�p��A��;*#
!
Environment/Episode Length  �B�Ϣ�)       7�_ 	�~}�p��A��;*

Losses/Value LossJݗAe�*       ����	��}�p��A��;*

Losses/Policy Loss��<�<֐,       ���E	��}�p��A��;*

Policy/Learning Rate�9�}�7       ���Y	�bw�q��A��<*(
&
Policy/Extrinsic Value EstimateDA�
��&       sO� 	��w�q��A��<*

Policy/EntropyO�?v��c5       ��]�	�w�q��A��<*&
$
Environment/Cumulative Reward((B!��2       $V�	,�w�q��A��<*#
!
Environment/Episode Length  �B�"��/       m]P	}�w�q��A��<* 

Policy/Extrinsic RewardV�BP��)       7�_ 	;�w�q��A��<*

Losses/Value Loss��A55�z*       ����	��w�q��A��<*

Losses/Policy LossT��<���,       ���E	��w�q��A��<*

Policy/Learning Rate"Í9kl�7       ���Y	�.B�r��A��=*(
&
Policy/Extrinsic Value Estimate��A�3�(/       m]P	/�B�r��A��=* 

Policy/Extrinsic Reward�� BT�)�&       sO� 	}�B�r��A��=*

Policy/Entropy�ί?�OT5       ��]�	��B�r��A��=*&
$
Environment/Cumulative RewardB�A2       $V�	��B�r��A��=*#
!
Environment/Episode Length  �B�Č�)       7�_ 	~�B�r��A��=*

Losses/Value Loss1��A��˟*       ����	��B�r��A��=*

Losses/Policy Loss��<4/i,       ���E	��B�r��A��=*

Policy/Learning Rate���9�zdn7       ���Y	��s��A��=*(
&
Policy/Extrinsic Value Estimate �Ag�k-/       m]P	��s��A��=* 

Policy/Extrinsic Reward��B���&       sO� 	f��s��A��=*

Policy/Entropy���?Aw�5       ��]�	'��s��A��=*&
$
Environment/Cumulative Reward�
B�	��2       $V�	��s��A��=*#
!
Environment/Episode Length  �B]Q��)       7�_ 	���s��A��=*

Losses/Value Loss.!�Ax�t*       ����	G��s��A��=*

Losses/Policy Loss�֫<��Hf,       ���E	)��s��A��=*

Policy/Learning Rate�n�9�(��7       ���Y	�o��t��A�>*(
&
Policy/Extrinsic Value Estimate�jA����&       sO� 	D���t��A�>*

Policy/Entropy�|�?l�N�5       ��]�	ڎ��t��A�>*&
$
Environment/Cumulative Rewardii	B-�:/2       $V�	Ü��t��A�>*#
!
Environment/Episode Length  �B6]�/       m]P	����t��A�>* 

Policy/Extrinsic Reward��B<���)       7�_ 	����t��A�>*

Losses/Value Loss�p�Am���*       ����	]���t��A�>*

Losses/Policy Loss�=�<�,       ���E		���t��A�>*

Policy/Learning Rate�C�9k�i�7       ���Y	�	�u��A��>*(
&
Policy/Extrinsic Value Estimatez`AG���/       m]P	KA	�u��A��>* 

Policy/Extrinsic RewardT�B�?V�&       sO� 	�C	�u��A��>*

Policy/Entropyfm�?�S�5       ��]�	�E	�u��A��>*&
$
Environment/Cumulative Reward��B����2       $V�	�G	�u��A��>*#
!
Environment/Episode Length  �Bka�)       7�_ 	�T	�u��A��>*

Losses/Value Lossw+�AFG��*       ����	f	�u��A��>*

Losses/Policy Loss���<���,       ���E	�v	�u��A��>*

Policy/Learning Rate�9���G7       ���Y	�U%�v��A��?*(
&
Policy/Extrinsic Value EstimateZ�%A��/       m]P	�%�v��A��?* 

Policy/Extrinsic Rewarddd$B��&       sO� 	�%�v��A��?*

Policy/Entropy�=�?� �05       ��]�	�%�v��A��?*&
$
Environment/Cumulative Reward��%B�	0�2       $V�	�%�v��A��?*#
!
Environment/Episode Length  �B�.�o)       7�_ 	ٕ%�v��A��?*

Losses/Value Loss��A���*       ����	��%�v��A��?*

Losses/Policy Loss�S�<#+��,       ���E	p�%�v��A��?*

Policy/Learning Ratef�9�(��7       ���Y	FC#�v��A��@*(
&
Policy/Extrinsic Value Estimate�YAG�P�&       sO� 	�G#�v��A��@*

Policy/Entropy���?0�5       ��]�	�I#�v��A��@*&
$
Environment/Cumulative Reward��B/:� 2       $V�	�K#�v��A��@*#
!
Environment/Episode Length  �B��=/       m]P	�Q#�v��A��@* 

Policy/Extrinsic Reward�lBne��7       ���Y	���w��A��@*(
&
Policy/Extrinsic Value Estimate�	 Aᬄ6/       m]P	���w��A��@* 

Policy/Extrinsic Reward��B��rJ)       7�_ 	���w��A��@*

Losses/Value Loss1�AD�Nw*       ����	���w��A��@*

Losses/Policy Loss*��<�>��,       ���E	�!��w��A��@*

Policy/Learning RateuŌ9�(�O&       sO� 	U3��w��A��@*

Policy/EntropyZ�?��!�5       ��]�	P7��w��A��@*&
$
Environment/Cumulative Reward��Bt�HP2       $V�	�;��w��A��@*#
!
Environment/Episode Length  �B-n`(7       ���Y	��x��A��A*(
&
Policy/Extrinsic Value Estimate��&Aan��/       m]P	Mi��x��A��A* 

Policy/Extrinsic Reward��B���&       sO� 	al��x��A��A*

Policy/Entropy,�?�uzs)       7�_ 	]n��x��A��A*

Losses/Value Loss���AW���*       ����	3p��x��A��A*

Losses/Policy Loss}Z�<p�,       ���E	6r��x��A��A*

Policy/Learning Rateښ�9���h5       ��]�	t��x��A��A*&
$
Environment/Cumulative Reward�JB�=�[2       $V�	qv��x��A��A*#
!
Environment/Episode Length  �B��7       ���Y	 c�y��A��A*(
&
Policy/Extrinsic Value Estimate�a3As=;&       sO� 	JAc�y��A��A*

Policy/EntropyL��?�l�5       ��]�	%Dc�y��A��A*&
$
Environment/Cumulative Reward��'B���`2       $V�	sFc�y��A��A*#
!
Environment/Episode Length  �BJ�5/       m]P	�Hc�y��A��A* 

Policy/Extrinsic Reward�)B�}�@)       7�_ 	�Jc�y��A��A*

Losses/Value Loss���A�%�*       ����	�Qc�y��A��A*

Losses/Policy Lossb��<��Ng,       ���E	�Sc�y��A��A*

Policy/Learning Rate�p�9�+��7       ���Y	rQ��z��A��B*(
&
Policy/Extrinsic Value Estimate_�2AX_�$/       m]P	�P��z��A��B* 

Policy/Extrinsic Reward tBg�}�&       sO� 	�U��z��A��B*

Policy/Entropy���?	7p5       ��]�	�Y��z��A��B*&
$
Environment/Cumulative Reward!B
��2       $V�	t\��z��A��B*#
!
Environment/Episode Length  �B8�)       7�_ 	x_��z��A��B*

Losses/Value Loss���AO�o�*       ����	9b��z��A��B*

Losses/Policy Lossj��<��R�,       ���E	nr��z��A��B*

Policy/Learning RateKF�9����7       ���Y	�Cl�{��A��C*(
&
Policy/Extrinsic Value Estimate6Ǎ�a/       m]P	�pl�{��A��C* 

Policy/Extrinsic RewardAA1Bڒ��&       sO� 	sl�{��A��C*

Policy/EntropyJ�?KǛ5       ��]�	�tl�{��A��C*&
$
Environment/Cumulative Reward  /B��Rs2       $V�	�vl�{��A��C*#
!
Environment/Episode Length  �B��fl)       7�_ 	Sxl�{��A��C*

Losses/Value Loss{��A@.�
*       ����	�yl�{��A��C*

Losses/Policy Loss���<O8+),       ���E	)�l�{��A��C*

Policy/Learning Rate[�9`��7       ���Y	 ca�|��A��C*(
&
Policy/Extrinsic Value Estimate�@A���&       sO� 	C�a�|��A��C*

Policy/Entropy�"�?"�5       ��]�	Þa�|��A��C*&
$
Environment/Cumulative Rewardss?B���2       $V�	�a�|��A��C*#
!
Environment/Episode Length  �Be�Qp/       m]P	��a�|��A��C* 

Policy/Extrinsic Rewardj@B�l]�)       7�_ 	X�a�|��A��C*

Losses/Value Loss�Q�AM�#�*       ����	x�a�|��A��C*

Losses/Policy Loss�3�<��M�,       ���E	��a�|��A��C*

Policy/Learning Rate��9n�G7       ���Y	��ް}��A��D*(
&
Policy/Extrinsic Value Estimate<�2A@�/       m]P	�ް}��A��D* 

Policy/Extrinsic Reward�QB�c�&       sO� 	��ް}��A��D*

Policy/Entropyyȭ?>��5       ��]�	��ް}��A��D*&
$
Environment/Cumulative Reward22B苪�2       $V�	u�ް}��A��D*#
!
Environment/Episode Length  �B��)       7�_ 	C�ް}��A��D*

Losses/Value Loss�GB���[*       ����	�߰}��A��D*

Losses/Policy Loss9��<\�cY,       ���E	A߰}��A��D*

Policy/Learning Rate�ǋ9ܤ̃7       ���Y	���~��A��D*(
&
Policy/Extrinsic Value Estimate|}<A��,�/       m]P	�Q��~��A��D* 

Policy/Extrinsic Reward#B8X�&       sO� 	�(��~��A��D*

Policy/Entropy��?Q~65       ��]�	]n��~��A��D*&
$
Environment/Cumulative RewardUU%B�<��2       $V�	����~��A��D*#
!
Environment/Episode Length  �B��z`)       7�_ 	���~��A��D*

Losses/Value Loss&��AèB�*       ����	9*��~��A��D*

Losses/Policy LossWڷ<�=|�,       ���E	�0��~��A��D*

Policy/Learning Rate.��9XC�7       ���Y	��Y���A��E*(
&
Policy/Extrinsic Value Estimate�0IAq�6&       sO� 	��Z���A��E*

Policy/Entropy�E�?�a.�5       ��]�	��Z���A��E*&
$
Environment/Cumulative Reward��8B�zk2       $V�	��Z���A��E*#
!
Environment/Episode Length  �B��/       m]P	
�Z���A��E* 

Policy/Extrinsic Reward�o8B���d)       7�_ 	��Z���A��E*

Losses/Value Loss�0�A+��*       ����	��Z���A��E*

Losses/Policy Loss&��<^��,       ���E	0�Z���A��E*

Policy/Learning Rate=s�9,��7       ���Y	0�⒀��A��F*(
&
Policy/Extrinsic Value EstimateM�BAŢ	/       m]P	-㒀��A��F* 

Policy/Extrinsic RewardJd.B!C��&       sO� 	z1㒀��A��F*

Policy/Entropy�ɬ?��'x5       ��]�	�7㒀��A��F*&
$
Environment/Cumulative Reward.B�5_�2       $V�	�;㒀��A��F*#
!
Environment/Episode Length  �B���)       7�_ 	VC㒀��A��F*

Losses/Value Loss�X�Ap��s*       ����	fK㒀��A��F*

Losses/Policy Loss���<ZT��,       ���E	{O㒀��A��F*

Policy/Learning Rate�H�9r�,�7       ���Y	�Շ���A��F*(
&
Policy/Extrinsic Value Estimateto\A��/       m]P	�eև���A��F* 

Policy/Extrinsic RewardUUUB���&       sO� 	,iև���A��F*

Policy/Entropy���?y���5       ��]�	3qև���A��F*&
$
Environment/Cumulative Reward�jWBRf��2       $V�	�~և���A��F*#
!
Environment/Episode Length  �B���T)       7�_ 	.�և���A��F*

Losses/Value Loss�Br�5x*       ����	[�և���A��F*

Losses/Policy Loss���<�1ǌ,       ���E	ݕև���A��F*

Policy/Learning Rate��9���37       ���Y	�i�{���AдG*(
&
Policy/Extrinsic Value Estimatea�SA��	�&       sO� 	v��{���AдG*

Policy/Entropy���?Ƶ�u5       ��]�	P��{���AдG*&
$
Environment/Cumulative Reward22HB��X2       $V�	*��{���AдG*#
!
Environment/Episode Length  �B�I4�/       m]P	⯐{���AдG* 

Policy/Extrinsic Reward�IB�/��)       7�_ 	���{���AдG*

Losses/Value Lossd�BjN*       ����	1��{���AдG*

Losses/Policy LossŔ�<�xc�,       ���E	���{���AдG*

Policy/Learning Rate�9�| �7       ���Y	���q���A��H*(
&
Policy/Extrinsic Value EstimateM�YA|c�/       m]P	�ߌq���A��H* 

Policy/Extrinsic Reward��?B��&       sO� 	��q���A��H*

Policy/EntropyZo�?���5       ��]�	��q���A��H*&
$
Environment/Cumulative RewardUUGB�Oak2       $V�	K�q���A��H*#
!
Environment/Episode Length  �B߳9N)       7�_ 	��q���A��H*

Losses/Value Loss��B�о�*       ����	B�q���A��H*

Losses/Policy LossI,�<Egz�,       ���E	���q���A��H*

Policy/Learning Rateʊ9� UH7       ���Y	���c���A��H*(
&
Policy/Extrinsic Value Estimate2�dA[?L/       m]P	���c���A��H* 

Policy/Extrinsic Reward--OB�֗3&       sO� 	M��c���A��H*

Policy/Entropy�@�?6�C�5       ��]�	���c���A��H*&
$
Environment/Cumulative Reward �HB<#92       $V�	 ��c���A��H*#
!
Environment/Episode Length  �Bg�E)       7�_ 	��c���A��H*

Losses/Value Loss��B�@�*       ����	"��c���A��H*

Losses/Policy Loss���<p�S,       ���E	8�c���A��H*

Policy/Learning Rate���9�mW�7       ���Y	zj+V���A��I*(
&
Policy/Extrinsic Value Estimate�yA*HB.&       sO� 	m�+V���A��I*

Policy/Entropy��?�/[5       ��]�	��+V���A��I*&
$
Environment/Cumulative Reward��sB�s�2       $V�	)�+V���A��I*#
!
Environment/Episode Length  �Ba/       m]P	'�+V���A��I* 

Policy/Extrinsic Reward�~rBΉa�)       7�_ 	<�+V���A��I*

Losses/Value Loss�BŦ�*       ����	S�+V���A��I*

Losses/Policy LossA��<�,       ���E	$�+V���A��I*

Policy/Learning Rate�u�9�['�7       ���Y	$��K���A��I*(
&
Policy/Extrinsic Value Estimate�fA�a�/       m]P	�ؙK���A��I* 

Policy/Extrinsic Reward�IHB��K�&       sO� 	1�K���A��I*

Policy/Entropy��?1K��5       ��]�	��K���A��I*&
$
Environment/Cumulative RewardddJBZ˪�2       $V�	��K���A��I*#
!
Environment/Episode Length  �Bd�v)       7�_ 	� �K���A��I*

Losses/Value Loss��B�7�*       ����	��K���A��I*

Losses/Policy Loss���<u#�,       ���E	��K���A��I*

Policy/Learning Rate�J�9�d��7       ���Y	���?���A��J*(
&
Policy/Extrinsic Value Estimate]�jA�M&�/       m]P	z��?���A��J* 

Policy/Extrinsic RewardSB�� d&       sO� 	m��?���A��J*

Policy/Entropy�o�?�B��5       ��]�	���?���A��J*&
$
Environment/Cumulative Reward��OB=�w2       $V�	`��?���A��J*#
!
Environment/Episode Length  �B�o�)       7�_ 	D��?���A��J*

Losses/Value Loss\�Bap�G*       ����	�?���A��J*

Losses/Policy Loss3�<Ry�,       ���E	�?���A��J*

Policy/Learning Rate!�9�_3�7       ���Y	���4���A��K*(
&
Policy/Extrinsic Value Estimate�hA�-S�&       sO� 	��4���A��K*

Policy/Entropy'�?e��5       ��]�	#�4���A��K*&
$
Environment/Cumulative RewardCB��2       $V�	��4���A��K*#
!
Environment/Episode Length  �B�+/       m]P	��4���A��K* 

Policy/Extrinsic Reward�gCB�H�)       7�_ 	��4���A��K*

Losses/Value Loss%�B�*       ����	�4���A��K*

Losses/Policy LossY$�<�� ,       ���E	��4���A��K*

Policy/Learning Ratec��9��7       ���Y	������A��K*(
&
Policy/Extrinsic Value Estimate�xAK���/       m]P	�����A��K* 

Policy/Extrinsic Reward˓TB�;ԡ&       sO� 	�����A��K*

Policy/Entropy2̪?|���5       ��]�	�
����A��K*&
$
Environment/Cumulative RewardRBNJ�z2       $V�	c����A��K*#
!
Environment/Episode Length  �B�҈�)       7�_ 	����A��K*

Losses/Value Loss�B&��F*       ����	�����A��K*

Losses/Policy Loss�v�<_P,       ���E	{����A��K*

Policy/Learning Rater̉9Р`�7       ���Y	T�0���AХL*(
&
Policy/Extrinsic Value Estimatex�zA��K/       m]P	^�0���AХL* 

Policy/Extrinsic Reward��GB >��&       sO� 	n�0���AХL*

Policy/EntropyӀ�?�ҡ5       ��]�	f�0���AХL*&
$
Environment/Cumulative Reward  JB��f2       $V�	D�0���AХL*#
!
Environment/Episode Length  �B��rL)       7�_ 	/�0���AХL*

Losses/Value Loss>B7N��*       ����	��0���AХL*

Losses/Policy LossH�<��@,       ���E	��0���AХL*

Policy/Learning Rateԡ�9�Z7       ���Y	�'EI���A��L*(
&
Policy/Extrinsic Value EstimateH�|A�v~�&       sO� 	�,EI���A��L*

Policy/Entropy�!�?jm�5       ��]�	�.EI���A��L*&
$
Environment/Cumulative RewardFFPB�l��2       $V�	�0EI���A��L*#
!
Environment/Episode Length  �B����/       m]P	3EI���A��L* 

Policy/Extrinsic RewardO�OB���7       ���Y	�#+5���A��M*(
&
Policy/Extrinsic Value EstimateV�A$Q�/       m]P	�f+5���A��M* 

Policy/Extrinsic RewardZ�xBF�#�)       7�_ 	�i+5���A��M*

Losses/Value Loss1jB�H�*       ����	�k+5���A��M*

Losses/Policy Loss��<U$N�,       ���E	�m+5���A��M*

Policy/Learning Rate�w�9����&       sO� 	�o+5���A��M*

Policy/Entropy���?�w5       ��]�	Oq+5���A��M*&
$
Environment/Cumulative Reward��wB��W�2       $V�	�s+5���A��M*#
!
Environment/Episode Length  �B\�7       ���Y	�N�,���A��N*(
&
Policy/Extrinsic Value EstimatedT�A���/       m]P	{n�,���A��N* 

Policy/Extrinsic RewardoB���&       sO� 	yq�,���A��N*

Policy/Entropys�?NZ.�)       7�_ 	E��,���A��N*

Losses/Value LossݩB;��*       ����	��,���A��N*

Losses/Policy LossTގ<V�O�,       ���E	A��,���A��N*

Policy/Learning RateEM�9��M�5       ��]�	X��,���A��N*&
$
Environment/Cumulative Reward�JoB���
2       $V�	S��,���A��N*#
!
Environment/Episode Length  �B�:�7       ���Y	������A��N*(
&
Policy/Extrinsic Value Estimate���A�8�9&       sO� 	!A����A��N*

Policy/Entropy��?��U�5       ��]�	E�����A��N*&
$
Environment/Cumulative Reward<<lB�~^�2       $V�	{2����A��N*#
!
Environment/Episode Length  �BSO/       m]P	8����A��N* 

Policy/Extrinsic Reward�<lB�� )       7�_ 	�;����A��N*

Losses/Value Loss!s B��#k*       ����	�>����A��N*

Losses/Policy Loss�n�<�D��,       ���E	�A����A��N*

Policy/Learning RateU#�9��&7       ���Y	�Q����A��O*(
&
Policy/Extrinsic Value Estimateb��AW^7V/       m]P	������A��O* 

Policy/Extrinsic Reward7�oB�r�&       sO� 	������A��O*

Policy/Entropy��??�$�5       ��]�	������A��O*&
$
Environment/Cumulative Reward

pB�-.Y2       $V�	������A��O*#
!
Environment/Episode Length  �B�{)       7�_ 	}�����A��O*

Losses/Value Loss<�KB�(�*       ����	c�����A��O*

Losses/Policy Loss��<vvI,       ���E	�����A��O*

Policy/Learning Rate���9/��7       ���Y	�������A��O*(
&
Policy/Extrinsic Value Estimate�_�Aԫ�%/       m]P	6�����A��O* 

Policy/Extrinsic Reward77sB��Y&       sO� 	�2�����A��O*

Policy/Entropy��?D"��5       ��]�	6�����A��O*&
$
Environment/Cumulative RewardU5sB֢��2       $V�	�8�����A��O*#
!
Environment/Episode Length  �Bā�)       7�_ 	q:�����A��O*

Losses/Value Loss|+B0���*       ����	;<�����A��O*

Losses/Policy Loss���<)Ə~,       ���E	�N�����A��O*

Policy/Learning Rate�Έ9J���7       ���Y	T�r�A��P*(
&
Policy/Extrinsic Value Estimate��Aq���&       sO� 	� s�A��P*

Policy/Entropy5j�?l;�5       ��]�	�#s�A��P*&
$
Environment/Cumulative Reward���B^3�2       $V�	�4s�A��P*#
!
Environment/Episode Length  �B��0V/       m]P	�7s�A��P* 

Policy/Extrinsic Reward���BL��)       7�_ 	�9s�A��P*

Losses/Value LossC�<BY@*       ����	q;s�A��P*

Losses/Policy Loss'T�<���,       ���E	GVs�A��P*

Policy/Learning Rate+��9~-�G7       ���Y	��Ԑ��AЖQ*(
&
Policy/Extrinsic Value Estimate�ףAq6�/       m]P	�8�Ԑ��AЖQ* 

Policy/Extrinsic Reward��nB`�o&       sO� 	}<�Ԑ��AЖQ*

Policy/Entropy�I�?���5       ��]�	�>�Ԑ��AЖQ*&
$
Environment/Cumulative Reward��oB��7I2       $V�	�@�Ԑ��AЖQ*#
!
Environment/Episode Length  �B�Hc�)       7�_ 	�N�Ԑ��AЖQ*

Losses/Value Loss
�NB���|*       ����	�b�Ԑ��AЖQ*

Losses/Policy Loss�<{���,       ���E	?s�Ԑ��AЖQ*

Policy/Learning Rate:z�9�sM7       ���Y	��w����A��Q*(
&
Policy/Extrinsic Value EstimategL�A��z/       m]P	��x����A��Q* 

Policy/Extrinsic Reward�Bad��&       sO� 	��x����A��Q*

Policy/Entropy["�?m���5       ��]�	��x����A��Q*&
$
Environment/Cumulative Reward ��B�"�2       $V�	P�x����A��Q*#
!
Environment/Episode Length  �B�~ue)       7�_ 	��x����A��Q*

Losses/Value Loss�m<Bn)+*       ����	$�x����A��Q*

Losses/Policy Loss��=)�,       ���E	��x����A��Q*

Policy/Learning Rate�O�9(�h�7       ���Y	�d䡒��A�R*(
&
Policy/Extrinsic Value Estimate�B�A3��^&       sO� 	��䡒��A�R*

Policy/Entropy�?|�P�5       ��]�	̵䡒��A�R*&
$
Environment/Cumulative Reward���B̵�2       $V�	��䡒��A�R*#
!
Environment/Episode Length  �B��.1/       m]P	��䡒��A�R* 

Policy/Extrinsic Rewardw��B�c�)       7�_ 	��䡒��A�R*

Losses/Value LossrRB��+�*       ����	��䡒��A�R*

Losses/Policy Lossڕ�<��,       ���E	�䡒��A�R*

Policy/Learning Rate�%�9"1*�7       ���Y	��,����A��S*(
&
Policy/Extrinsic Value EstimateIA�A\���/       m]P	G�-����A��S* 

Policy/Extrinsic Reward��B���&       sO� 	��-����A��S*

Policy/Entropy�ɧ?�Lt5       ��]�	��-����A��S*&
$
Environment/Cumulative Rewardii�Bjj�^2       $V�	w�-����A��S*#
!
Environment/Episode Length  �BX�A�)       7�_ 	�	.����A��S*

Losses/Value Loss��SB.0G�*       ����	�.����A��S*

Losses/Policy Lossej�<mr��,       ���E	.V.����A��S*

Policy/Learning Rate��9FX��7       ���Y	�%�y���A��S*(
&
Policy/Extrinsic Value Estimate�y�A�\�./       m]P	<��y���A��S* 

Policy/Extrinsic Reward77�BZ@�&       sO� 	��y���A��S*

Policy/Entropy���?�a 5       ��]�	�óy���A��S*&
$
Environment/Cumulative RewardU��Br�?2       $V�	�ǳy���A��S*#
!
Environment/Episode Length  �B	>�)       7�_ 	�ɳy���A��S*

Losses/Value Losse-SBo�w*       ����	�˳y���A��S*

Losses/Policy Loss���<��f+,       ���E	�ڳy���A��S*

Policy/Learning Rateч9�E�7       ���Y	���\���A��T*(
&
Policy/Extrinsic Value Estimate
ͽA�"�m&       sO� 	\�\���A��T*

Policy/Entropy�|�?�Bw5       ��]�	?q�\���A��T*&
$
Environment/Cumulative Reward���B;D:�2       $V�	2u�\���A��T*#
!
Environment/Episode Length  �B֘B&/       m]P	�w�\���A��T* 

Policy/Extrinsic Reward?[�Bv���)       7�_ 	���\���A��T*

Losses/Value Loss��VB#�0 *       ����	D��\���A��T*

Losses/Policy Loss�w�<
?E,       ���E	d��\���A��T*

Policy/Learning Rate~��9� �7       ���Y	Q��K���A��T*(
&
Policy/Extrinsic Value Estimate2ڿAN�[�/       m]P	C�K���A��T* 

Policy/Extrinsic Reward?�B�ڵ�&       sO� 	��K���A��T*

Policy/Entropy�*�?��b5       ��]�	`!�K���A��T*&
$
Environment/Cumulative Reward�B/'�2       $V�	�#�K���A��T*#
!
Environment/Episode Length  �B�	��)       7�_ 	&�K���A��T*

Losses/Value Loss3�RB���*       ����	4�K���A��T*

Losses/Policy Loss1�<v�,       ���E	�6�K���A��T*

Policy/Learning Rate�|�9���7       ���Y	��6���A��U*(
&
Policy/Extrinsic Value Estimate辽A��/       m]P	�"�6���A��U* 

Policy/Extrinsic Reward��B����&       sO� 	&�6���A��U*

Policy/Entropy��?�py�5       ��]�	�'�6���A��U*&
$
Environment/Cumulative Reward  �B-�A�2       $V�	�)�6���A��U*#
!
Environment/Episode Length  �BV-��)       7�_ 	�+�6���A��U*

Losses/Value Loss��BBB9*       ����	�8�6���A��U*

Losses/Policy Loss���<�I,       ���E	X;�6���A��U*

Policy/Learning Rate�Q�9Q3�7       ���Y	40���AЇV*(
&
Policy/Extrinsic Value Estimate��AI%0&       sO� 	�s0���AЇV*

Policy/Entropy���?�ޤS5       ��]�	�w0���AЇV*&
$
Environment/Cumulative Reward���B� ��2       $V�	�y0���AЇV*#
!
Environment/Episode Length  �B���/       m]P	�{0���AЇV* 

Policy/Extrinsic Reward�<�B%u�7)       7�_ 	F}0���AЇV*

Losses/Value Loss�#iB[;��*       ����	�~0���AЇV*

Losses/Policy Loss���<�j�,       ���E	��0���AЇV*

Policy/Learning Rate�'�96��7       ���Y	K	~���A��V*(
&
Policy/Extrinsic Value Estimate��A��/       m]P	g~���A��V* 

Policy/Extrinsic RewardE��B�OY&       sO� 	�i~���A��V*

Policy/Entropy�m�?�&F<5       ��]�	�l~���A��V*&
$
Environment/Cumulative Reward���Ba�rh2       $V�	�{~���A��V*#
!
Environment/Episode Length  �Bٺ��)       7�_ 	�~���A��V*

Losses/Value Loss�{;B��
�*       ����	��~���A��V*

Losses/Policy Loss+��<~�,       ���E	s�~���A��V*

Policy/Learning Rate`��9M�P�7       ���Y	ᓣ�A�W*(
&
Policy/Extrinsic Value Estimatejw�A��|/       m]P	9��A�W* 

Policy/Extrinsic Reward�B��d&       sO� 	=��A�W*

Policy/Entropy#�?oL�5       ��]�	���A�W*&
$
Environment/Cumulative Reward��B޶��2       $V�	���A�W*#
!
Environment/Episode Length  �Bh�&)       7�_ 	���A�W*

Losses/Value Loss�jZB�:Yc*       ����	���A�W*

Losses/Policy Loss�<6�l�,       ���E	j��A�W*

Policy/Learning Rateoӆ9rF(7       ���Y	Yl�ޚ��A��W*(
&
Policy/Extrinsic Value Estimate��AoR<�&       sO� 	̙�ޚ��A��W*

Policy/Entropy�
�?;�]�5       ��]�	���ޚ��A��W*&
$
Environment/Cumulative RewardPP�B�G�2       $V�	��ޚ��A��W*#
!
Environment/Episode Length  �B~VM_/       m]P	���ޚ��A��W* 

Policy/Extrinsic Reward�y�BM0�)       7�_ 	6��ޚ��A��W*

Losses/Value Loss�VhB����*       ����	=��ޚ��A��W*

Losses/Policy Lossy��<��m,       ���E	��ޚ��A��W*

Policy/Learning RateѨ�9�΂�7       ���Y	*V�Û��A��X*(
&
Policy/Extrinsic Value Estimate���A�&�]/       m]P	�e�Û��A��X* 

Policy/Extrinsic Reward�Q�B0u<&       sO� 	h�Û��A��X*

Policy/Entropy��?ڊ�5       ��]�	�i�Û��A��X*&
$
Environment/Cumulative Rewardnn�B)��2       $V�	�k�Û��A��X*#
!
Environment/Episode Length  �B�l8)       7�_ 	�m�Û��A��X*

Losses/Value Loss��WB���*       ����	Yo�Û��A��X*

Losses/Policy Loss���<^��,       ���E	Xr�Û��A��X*

Policy/Learning Rate�~�9Bq��7       ���Y	�7�����A��Y*(
&
Policy/Extrinsic Value Estimate.��A��z�/       m]P	T�����A��Y* 

Policy/Extrinsic Rewardnn�B�3:]&       sO� 	�V�����A��Y*

Policy/Entropy��?1Ã�5       ��]�	�X�����A��Y*&
$
Environment/Cumulative Reward ��B���E2       $V�	�[�����A��Y*#
!
Environment/Episode Length  �BZ�Ҫ)       7�_ 	(_�����A��Y*

Losses/Value Loss��^B=�|�*       ����	|a�����A��Y*

Losses/Policy Lossw��<	|V,       ���E	^c�����A��Y*

Policy/Learning RateBT�9z��7       ���Y	��D眨�A��Y*(
&
Policy/Extrinsic Value Estimate���A����&       sO� 	��D眨�A��Y*

Policy/Entropy�n�?�\V5       ��]�	L�D眨�A��Y*&
$
Environment/Cumulative Reward���B��W2       $V�	��D眨�A��Y*#
!
Environment/Episode Length  �BQF7/       m]P	}E眨�A��Y* 

Policy/Extrinsic Reward��B	�+7       ���Y	�n�Ν��A��Z*(
&
Policy/Extrinsic Value Estimate���A_��/       m]P	Q��Ν��A��Z* 

Policy/Extrinsic Reward�~�BM�S)       7�_ 	}��Ν��A��Z*

Losses/Value Loss�YB��u*       ����	c��Ν��A��Z*

Losses/Policy Loss.@�<�.�,       ���E	���Ν��A��Z*

Policy/Learning RateR*�9�w&       sO� 	o��Ν��A��Z*

Policy/Entropy6S�?�Y��5       ��]�	I��Ν��A��Z*&
$
Environment/Cumulative Reward���BM�h2       $V�	��Ν��A��Z*#
!
Environment/Episode Length  �B�Ɂn7       ���Y	(^�����A��Z*(
&
Policy/Extrinsic Value Estimate�,�AɱC�/       m]P	������A��Z* 

Policy/Extrinsic RewardUU�Bb� �&       sO� 	+6�����A��Z*

Policy/Entropy��?��)       7�_ 	;�����A��Z*

Losses/Value LossݚPB����*       ����	�=�����A��Z*

Losses/Policy LossO �<����,       ���E	�?�����A��Z*

Policy/Learning Rate���9SLΞ5       ��]�	WA�����A��Z*&
$
Environment/Cumulative Reward��B��V2       $V�	 C�����A��Z*#
!
Environment/Episode Length  �Ba�su7       ���Y	�'����A��[*(
&
Policy/Extrinsic Value Estimate /�A@$��&       sO� 	a����A��[*

Policy/Entropyv٤?����5       ��]�	td����A��[*&
$
Environment/Cumulative Reward�͚BQ1d�2       $V�	�f����A��[*#
!
Environment/Episode Length  �Bdl/       m]P	Di����A��[* 

Policy/Extrinsic Reward��B�j5)       7�_ 	�o����A��[*

Losses/Value Loss�QB��*       ����	s����A��[*

Losses/Policy Loss�2�<R.? ,       ���E	�u����A��[*

Policy/Learning Rate�Յ9oMۋ7       ���Y	j�4����A�\*(
&
Policy/Extrinsic Value Estimates�A�o�Y/       m]P	��4����A�\* 

Policy/Extrinsic Reward���B!�&       sO� 	.�4����A�\*

Policy/Entropy�?/N��5       ��]�	�4����A�\*&
$
Environment/Cumulative Reward�ÔB��>22       $V�	��4����A�\*#
!
Environment/Episode Length  �B�A�)       7�_ 	x�4����A�\*

Losses/Value Loss�ZBP��*       ����	)�4����A�\*

Losses/Policy Loss��<ksk,       ���E	�4����A�\*

Policy/Learning Rate$��9����7       ���Y	ɯh���A��\*(
&
Policy/Extrinsic Value Estimate���A��.�/       m]P	`h���A��\* 

Policy/Extrinsic Reward���B�?�w&       sO� 	�	h���A��\*

Policy/Entropy�U�?�Q;G5       ��]�	h���A��\*&
$
Environment/Cumulative Reward 0�B_��2       $V�	Rh���A��\*#
!
Environment/Episode Length  �B`�f)       7�_ 	�h���A��\*

Losses/Value Loss� bB�5ڰ*       ����	.h���A��\*

Losses/Policy Loss^(�<��֠,       ���E	 h���A��\*

Policy/Learning Rate4��9�2��7       ���Y	��WQ���A��]*(
&
Policy/Extrinsic Value EstimateqL�A�hS.&       sO� 	�"XQ���A��]*

Policy/Entropy�?��Hr5       ��]�	�%XQ���A��]*&
$
Environment/Cumulative RewardFF�B5 �r2       $V�	�'XQ���A��]*#
!
Environment/Episode Length  �BA�I�/       m]P	0)XQ���A��]* 

Policy/Extrinsic Reward��B𕸶)       7�_ 	�*XQ���A��]*

Losses/Value Loss�`B��ف*       ����	s,XQ���A��]*

Losses/Policy Loss��<3�(�,       ���E	�7XQ���A��]*

Policy/Learning Rate�V�9��.7       ���Y	R�9���A��]*(
&
Policy/Extrinsic Value Estimate��Ac�-�/       m]P	���9���A��]* 

Policy/Extrinsic Reward�4�B",&       sO� 	��9���A��]*

Policy/Entropy��?�u5       ��]�	^��9���A��]*&
$
Environment/Cumulative Rewardxx�Bc/�
2       $V�	��9���A��]*#
!
Environment/Episode Length  �BP�y)       7�_ 	���9���A��]*

Losses/Value LossKiPB�r�*       ����	���9���A��]*

Losses/Policy Lossy>�<�/a,       ���E	���9���A��]*

Policy/Learning Rate�,�9d���7       ���Y	\�O%���A��^*(
&
Policy/Extrinsic Value Estimate��A�z]/       m]P	��O%���A��^* 

Policy/Extrinsic Reward���B�D��&       sO� 	3�O%���A��^*

Policy/Entropy�أ?�4��5       ��]�	!�O%���A��^*&
$
Environment/Cumulative Reward�ʘBT9�2       $V�	� P%���A��^*#
!
Environment/Episode Length  �B6�dM)       7�_ 	�P%���A��^*

Losses/Value Loss%`BF[��*       ����	qP%���A��^*

Losses/Policy Loss���<�8JG,       ���E	P%���A��^*

Policy/Learning Rate�92���7       ���Y	1U���A��_*(
&
Policy/Extrinsic Value EstimateHo�A��L�&       sO� 	�U���A��_*

Policy/Entropy���?n��~5       ��]�	�U���A��_*&
$
Environment/Cumulative Reward���B���42       $V�	��U���A��_*#
!
Environment/Episode Length  �B:���/       m]P	�U���A��_* 

Policy/Extrinsic Rewardj�B2��)       7�_ 	�U���A��_*

Losses/Value Loss�yNBN��%*       ����	��U���A��_*

Losses/Policy Loss��<t�pq,       ���E	S�U���A��_*

Policy/Learning Rate؄9�2�7       ���Y	J���A��_*(
&
Policy/Extrinsic Value Estimate� B�=cm/       m]P	����A��_* 

Policy/Extrinsic Reward��BU�*&       sO� 	����A��_*

Policy/EntropyQg�?z�5       ��]�	����A��_*&
$
Environment/Cumulative Reward���B�A�2       $V�	���A��_*#
!
Environment/Episode Length  �B΢�)       7�_ 	]���A��_*

Losses/Value Loss$�MB����*       ����	���A��_*

Losses/Policy Loss)��<&��2,       ���E	��A��_*

Policy/Learning Rate{��9}3Z7       ���Y	d��ئ��A�`*(
&
Policy/Extrinsic Value EstimateO� B���/       m]P	��ئ��A�`* 

Policy/Extrinsic Reward���Bhz��&       sO� 	��ئ��A�`*

Policy/Entropyq��?�*5       ��]�	��ئ��A�`*&
$
Environment/Cumulative Reward�ʓB	>%k2       $V�	h�ئ��A�`*#
!
Environment/Episode Length  �B�]x�)       7�_ 	-
�ئ��A�`*

Losses/Value Loss��[B�.��*       ����	��ئ��A�`*

Losses/Policy Lossg�<��J�,       ���E	��ئ��A�`*

Policy/Learning Rate���9�7       ���Y	Y�p����A��a*(
&
Policy/Extrinsic Value Estimate(XB�c&       sO� 	�vq����A��a*

Policy/EntropyѨ�?���5       ��]�	�yq����A��a*&
$
Environment/Cumulative Reward�BB�Y�2       $V�	�}q����A��a*#
!
Environment/Episode Length  �B��Q/       m]P	�q����A��a* 

Policy/Extrinsic Reward��Bv�W)       7�_ 	f�q����A��a*

Losses/Value LossbYB�/*       ����	�q����A��a*

Losses/Policy Loss��<�%e,       ���E	ݒq����A��a*

Policy/Learning Rate�X�9�*�C7       ���Y	Ho�����A��a*(
&
Policy/Extrinsic Value Estimate��	B?�m/       m]P	@ݱ����A��a* 

Policy/Extrinsic Reward˓�BZ$�&       sO� 	�౬���A��a*

Policy/EntropyH��?L#��5       ��]�	�ⱬ���A��a*&
$
Environment/Cumulative RewardKK�B�zo2       $V�	�䱬���A��a*#
!
Environment/Episode Length  �B�~x�)       7�_ 	�汬���A��a*

Losses/Value Loss<�RB��sG*       ����	�豬���A��a*

Losses/Policy Loss+�<5�.�,       ���E	F걬���A��a*

Policy/Learning Rate�.�9�E �7       ���Y	�{����A��b*(
&
Policy/Extrinsic Value Estimate)�B��0/       m]P	�����A��b* 

Policy/Extrinsic Reward�BMJ�&       sO� 	
�����A��b*

Policy/Entropy-h�?�Fm5       ��]�	������A��b*&
$
Environment/Cumulative Reward �B D	�2       $V�	! ����A��b*#
!
Environment/Episode Length  �B���))       7�_ 	Z����A��b*

Losses/Value Loss�(UB��Y-*       ����	�����A��b*

Losses/Policy Loss~0�<�8��,       ���E	����A��b*

Policy/Learning Rate]�9�` 7       ���Y	�@z���A��b*(
&
Policy/Extrinsic Value Estimatem�B��O�&       sO� 	T�z���A��b*

Policy/Entropy�:�?��G�5       ��]�	��z���A��b*&
$
Environment/Cumulative Reward  �B 	��2       $V�	��z���A��b*#
!
Environment/Episode Length  �B�ď�/       m]P	�z���A��b* 

Policy/Extrinsic Reward���Be�)       7�_ 	ɯz���A��b*

Losses/Value LossS)UBG>)�*       ����	F�z���A��b*

Losses/Policy Loss�q�<ķm�,       ���E	�z���A��b*

Policy/Learning Ratelڃ9�[�R7       ���Y	��{c���A��c*(
&
Policy/Extrinsic Value Estimate��B�Ɖ�/       m]P	�|c���A��c* 

Policy/Extrinsic Rewardb��BZ���&       sO� 	0|c���A��c*

Policy/Entropy�?��o5       ��]�	&|c���A��c*&
$
Environment/Cumulative Reward��B���j2       $V�	�$|c���A��c*#
!
Environment/Episode Length  �BZ�r@)       7�_ 	�(|c���A��c*

Losses/Value Loss>MB��*       ����	�*|c���A��c*

Losses/Policy Lossd/�<���,       ���E	�,|c���A��c*

Policy/Learning Rateί�9[�7       ���Y	��F���A��d*(
&
Policy/Extrinsic Value EstimateG�BuFȑ/       m]P	�7�F���A��d* 

Policy/Extrinsic RewardPP�B��&       sO� 	h=�F���A��d*

Policy/Entropy���?���"5       ��]�	�?�F���A��d*&
$
Environment/Cumulative RewardU��B��rL2       $V�	�A�F���A��d*#
!
Environment/Episode Length  �B+R��)       7�_ 	yC�F���A��d*

Losses/Value LossǩeBj� �*       ����	AE�F���A��d*

Losses/Policy Loss<�<�=p�,       ���E	G�F���A��d*

Policy/Learning Rateޅ�9̊��7       ���Y	�DY2���A��d*(
&
Policy/Extrinsic Value Estimate��Bl�&       sO� 	�vY2���A��d*

Policy/Entropy.��?<�͒5       ��]�	��Y2���A��d*&
$
Environment/Cumulative Reward�B�6,J2       $V�	f�Y2���A��d*#
!
Environment/Episode Length  �B7�z1/       m]P	m�Y2���A��d* 

Policy/Extrinsic Reward��BE���)       7�_ 	7�Y2���A��d*

Losses/Value Loss�ҞBM��*       ����	�Y2���A��d*

Losses/Policy Loss%ԣ<���S,       ���E	��Y2���A��d*

Policy/Learning Rate?[�9@w"7       ���Y	��N���A�e*(
&
Policy/Extrinsic Value Estimate03(Bw��/       m]P	O���A�e* 

Policy/Extrinsic Reward7˩B���&       sO� 	l&O���A�e*

Policy/Entropyb�?5��5       ��]�	9*O���A�e*&
$
Environment/Cumulative RewardUU�B�\�2       $V�	0,O���A�e*#
!
Environment/Episode Length  �B�Q�)       7�_ 	.O���A�e*

Losses/Value Loss�6B�Fx*       ����	�/O���A�e*

Losses/Policy Loss��<	U�,       ���E	�1O���A�e*

Policy/Learning RateN1�9�w�H7       ���Y	7ii���A��e*(
&
Policy/Extrinsic Value Estimate��)B����/       m]P	�xi���A��e* 

Policy/Extrinsic Rewardnn�B���&       sO� 	�zi���A��e*

Policy/EntropyM�?��L�5       ��]�	}i���A��e*&
$
Environment/Cumulative Reward�*�B�c�2       $V�	i���A��e*#
!
Environment/Episode Length  �B~��B)       7�_ 	�i���A��e*

Losses/Value Losss�2B����*       ����	�i���A��e*

Losses/Policy Loss{�<�.?�,       ���E	��i���A��e*

Policy/Learning Rate��9p6�(7       ���Y	4a�>���A��f*(
&
Policy/Extrinsic Value Estimate*�/B��+&       sO� 	�k�>���A��f*

Policy/Entropy�Š?���5       ��]�	�o�>���A��f*&
$
Environment/Cumulative RewardPP�BY�z�2       $V�	�r�>���A��f*#
!
Environment/Episode Length  �B�]�/       m]P	Tv�>���A��f* 

Policy/Extrinsic Reward���B��w�7       ���Y	�(�)���A��g*(
&
Policy/Extrinsic Value Estimate�u.B���O/       m]P	�H�)���A��g* 

Policy/Extrinsic Reward˓�BC��})       7�_ 	
K�)���A��g*

Losses/Value Loss��?B��3 *       ����	S]�)���A��g*

Losses/Policy Loss���<�-~�,       ���E	Fa�)���A��g*

Policy/Learning Rate�܂9x���&       sO� 	kc�)���A��g*

Policy/Entropy���? �555       ��]�	Ve�)���A��g*&
$
Environment/Cumulative Reward77�B���2       $V�	'g�)���A��g*#
!
Environment/Episode Length  �B=M�+7       ���Y	������A��g*(
&
Policy/Extrinsic Value EstimateD�0B/+m(/       m]P	�t����A��g* 

Policy/Extrinsic Reward���B��C�&       sO� 	������A��g*

Policy/Entropy���?�D�)       7�_ 	������A��g*

Losses/Value Loss�7)B�	n�*       ����	u�����A��g*

Losses/Policy Loss���<����,       ���E	�����A��g*

Policy/Learning Rate"��9`2�{5       ��]�	ʤ����A��g*&
$
Environment/Cumulative Reward���B ��2       $V�	¦����A��g*#
!
Environment/Episode Length  �B�R�7       ���Y	�������A��h*(
&
Policy/Extrinsic Value Estimate3�6B�oݴ&       sO� 	�˚����A��h*

Policy/Entropy2|�? �I:5       ��]�	�͚����A��h*&
$
Environment/Cumulative Reward�ҦB���2       $V�	�Ϛ����A��h*#
!
Environment/Episode Length  �B�Ƈ/       m]P	�њ����A��h* 

Policy/Extrinsic Reward��B�Z��)       7�_ 	NӚ����A��h*

Losses/Value Loss�g3B/f�K*       ����	՚����A��h*

Losses/Policy Loss���<<��j,       ���E	8ؚ����A��h*

Policy/Learning Rate1��9?8p7       ���Y	�u鲨�A��h*(
&
Policy/Extrinsic Value Estimate�?9BC �/       m]P	dVv鲨�A��h* 

Policy/Extrinsic Reward�i�B�g�&       sO� 	dYv鲨�A��h*

Policy/Entropy<��?f�-5       ��]�	\[v鲨�A��h*&
$
Environment/Cumulative RewardUU�Bn�ʰ2       $V�	5]v鲨�A��h*#
!
Environment/Episode Length  �B��)       7�_ 	_v鲨�A��h*

Losses/Value Lossϓ%B�8'*       ����	�`v鲨�A��h*

Losses/Policy Loss�W�<`3��,       ���E	0cv鲨�A��h*

Policy/Learning Rate�]�9]�D�7       ���Y	Y��ҳ��A��i*(
&
Policy/Extrinsic Value Estimatev7BN��k/       m]P	vo�ҳ��A��i* 

Policy/Extrinsic Reward--�B���V&       sO� 	v�ҳ��A��i*

Policy/Entropy�}�?�t$$5       ��]�	�z�ҳ��A��i*&
$
Environment/Cumulative Reward ЩBB�r�2       $V�	�~�ҳ��A��i*#
!
Environment/Episode Length  �BR��)       7�_ 	���ҳ��A��i*

Losses/Value Loss"1B�kκ*       ����	���ҳ��A��i*

Losses/Policy Loss���<����,       ���E	��ҳ��A��i*

Policy/Learning Rate�3�9\�f7       ���Y	nj�´��A��j*(
&
Policy/Extrinsic Value EstimatedH8B�He&       sO� 	eĻ´��A��j*

Policy/Entropyg�?��5       ��]�	�ǻ´��A��j*&
$
Environment/Cumulative Reward���B(fKO2       $V�	�ɻ´��A��j*#
!
Environment/Episode Length  �B'��/       m]P	T˻´��A��j* 

Policy/Extrinsic Reward&�B��R)       7�_ 	ͻ´��A��j*

Losses/Value LossDX)B�[�*       ����	�λ´��A��j*

Losses/Policy Loss�<�<L�N�,       ���E	�л´��A��j*

Policy/Learning Rate	�9,j7       ���Y	#�"����A��j*(
&
Policy/Extrinsic Value Estimate�=B�Sę/       m]P	�&#����A��j* 

Policy/Extrinsic Reward��B�"�&       sO� 	�*#����A��j*

Policy/Entropy:T�?��V#5       ��]�	�,#����A��j*&
$
Environment/Cumulative Reward�ͫB�0	Q2       $V�	�.#����A��j*#
!
Environment/Episode Length  �B�4#�)       7�_ 	�0#����A��j*

Losses/Value Loss�"B�2*       ����	]2#����A��j*

Losses/Policy Loss���<52�C,       ���E	z4#����A��j*

Policy/Learning Rate߁9�]�%7       ���Y	/N�����A��k*(
&
Policy/Extrinsic Value Estimate��>B� �/       m]P	Dp�����A��k* 

Policy/Extrinsic Reward���B�І	&       sO� 	�t�����A��k*

Policy/Entropy�]�?~75       ��]�	 x�����A��k*&
$
Environment/Cumulative Reward�j�B�M*�2       $V�	{�����A��k*#
!
Environment/Episode Length  �B8ĺ)       7�_ 	#������A��k*

Losses/Value Loss� (B#�J�*       ����	�������A��k*

Losses/Policy Loss�B�<�F��,       ���E	������A��k*

Policy/Learning Rateu��9�$M7       ���Y	7܂���A��l*(
&
Policy/Extrinsic Value Estimate4�9B����&       sO� 	�~܂���A��l*

Policy/EntropyI�?$G��5       ��]�	b�܂���A��l*&
$
Environment/Cumulative RewardKK�B���%2       $V�	��܂���A��l*#
!
Environment/Episode Length  �Bռ�o/       m]P	��܂���A��l* 

Policy/Extrinsic Reward��Byq�)       7�_ 	�܂���A��l*

Losses/Value Loss"�BN\*       ����	3�܂���A��l*

Losses/Policy Loss��<g��,       ���E	��܂���A��l*

Policy/Learning Rate���9�HG7       ���Y	�	x���A��l*(
&
Policy/Extrinsic Value Estimate��>B	�KA/       m]P	�-	x���A��l* 

Policy/Extrinsic Reward-O�B��̳&       sO� 	�/	x���A��l*

Policy/Entropyl%�?]�5       ��]�	�1	x���A��l*&
$
Environment/Cumulative Reward��Bx�K�2       $V�	�3	x���A��l*#
!
Environment/Episode Length  �B�ª)       7�_ 	~5	x���A��l*

Losses/Value Loss�$B�/F*       ����	T7	x���A��l*

Losses/Policy Loss�S�<��,       ���E	:	x���A��l*

Policy/Learning Rate�_�9����7       ���Y	A��`���A��m*(
&
Policy/Extrinsic Value Estimate��DB6�PR/       m]P	�C�`���A��m* 

Policy/Extrinsic RewardKK�B��`&       sO� 	�G�`���A��m*

Policy/Entropy���?��\d5       ��]�	�I�`���A��m*&
$
Environment/Cumulative Reward �B\�&2       $V�	�K�`���A��m*#
!
Environment/Episode Length  �B����)       7�_ 	�\�`���A��m*

Losses/Value Loss]�(BX��*       ����	g`�`���A��m*

Losses/Policy Lossql�<�lߦ,       ���E	�b�`���A��m*

Policy/Learning Rate�5�9�`f�7       ���Y	�[�K���A��m*(
&
Policy/Extrinsic Value Estimate�:FBIG�&       sO� 	OǟK���A��m*

Policy/Entropyџ?�G�5       ��]�	�ٟK���A��m*&
$
Environment/Cumulative Reward���Bؙ L2       $V�	�۟K���A��m*#
!
Environment/Episode Length  �B��/       m]P	�ݟK���A��m* 

Policy/Extrinsic Reward߈�B�::A)       7�_ 	�ߟK���A��m*

Losses/Value Loss�GB����*       ����	X�K���A��m*

Losses/Policy LossI9�<�n@�,       ���E	��K���A��m*

Policy/Learning Rate[�9�'�7       ���Y	�|3���Aмn*(
&
Policy/Extrinsic Value Estimateg�CBc��/       m]P	�|3���Aмn* 

Policy/Extrinsic Reward��BWb&       sO� 	��|3���Aмn*

Policy/Entropy˲�?���5       ��]�	$�|3���Aмn*&
$
Environment/Cumulative Rewardss�BvΫ2       $V�	�|3���Aмn*#
!
Environment/Episode Length  �Bb�f�)       7�_ 	�|3���Aмn*

Losses/Value Loss>%!B8Ӄ�*       ����	��|3���Aмn*

Losses/Policy Loss���<�K��,       ���E	��|3���Aмn*

Policy/Learning Ratef�9���y7       ���Y	��< ���A��o*(
&
Policy/Extrinsic Value Estimate��EB���/       m]P	G�< ���A��o* 

Policy/Extrinsic Reward��BM�I�&       sO� 	6�< ���A��o*

Policy/Entropys��?'�4�5       ��]�	m�< ���A��o*&
$
Environment/Cumulative Reward  �B/<޶2       $V�	��< ���A��o*#
!
Environment/Episode Length  �BŶ*u)       7�_ 	\�< ���A��o*

Losses/Value Loss�B]X]*       ����	`�< ���A��o*

Losses/Policy Loss���<K��,       ���E	��< ���A��o*

Policy/Learning Rate˶�9Z+��7       ���Y	������A��o*(
&
Policy/Extrinsic Value Estimate	�CB�-R&       sO� 	"����A��o*

Policy/EntropyɄ�?�/E�5       ��]�	�*����A��o*&
$
Environment/Cumulative Reward77�B)�Y2       $V�	X:����A��o*#
!
Environment/Episode Length  �BU�@/       m]P	q?����A��o* 

Policy/Extrinsic RewardV��B��N)       7�_ 	B����A��o*

Losses/Value Lossv<B%���*       ����	o����A��o*

Losses/Policy LossU_�<?a�,       ���E	u����A��o*

Policy/Learning Rateی�9$Z�b