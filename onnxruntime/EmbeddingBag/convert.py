import sys
import math
import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.onnx import OperatorExportTypes

cur_maxsize= sys.maxsize
cur_sqrt_maxsize = float(math.floor(math.sqrt(cur_maxsize)))
cur_num_categories = int(cur_sqrt_maxsize)
cur_num_collisions = math.floor(math.sqrt(math.sqrt(cur_maxsize)))
print("maxsize: ", cur_maxsize)
print("sqrt_maxsize: ", cur_sqrt_maxsize)
print("num_categories: ", cur_num_categories)
print("num_collisions: ", cur_num_collisions)
cur_max_seq_length = 512
cur_padding_id = 0

class QREmbeddingBag(nn.Module):
    def __init__(self, num_categories=0, num_collisions=0, embedding_dim=0, padding_id=0, operation="mult", max_norm=None, norm_type=2, scale_grad_by_freq=False, mode="sum", sparse=False, _weight=None,):
        super(QREmbeddingBag, self).__init__()
        self.num_categories = int(num_categories)
        self.num_collisions = int(num_collisions)
        if isinstance(embedding_dim, int) or len(embedding_dim) == 1:
            self.embedding_dim = [embedding_dim, embedding_dim]
        else:
            self.embedding_dim = embedding_dim
        self.padding_id = padding_id
        self.operation = operation
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq

        if self.operation == "add" or self.operation == "mult":
            assert self.embedding_dim[0] == self.embedding_dim[1], "Embedding dimensions do not match!"

        self.num_embeddings = [int(np.ceil(self.num_categories / self.num_collisions)), self.num_collisions]

        if _weight is None:
            self.weight_q = Parameter(torch.Tensor(self.num_embeddings[0], self.embedding_dim[0]))
            self.weight_r = Parameter(torch.Tensor(self.num_embeddings[1], self.embedding_dim[1]))
            self.reset_parameters()
        else:
            assert list(_weight[0].shape) == [
                self.num_embeddings[0],
                self.embedding_dim[0],
            ], "Shape of weight for quotient table does not match num_embeddings and embedding_dim"
            assert list(_weight[1].shape) == [
                self.num_embeddings[1],
                self.embedding_dim[1],
            ], "Shape of weight for remainder table does not match num_embeddings and embedding_dim"
            self.weight_q = Parameter(_weight[0])
            self.weight_r = Parameter(_weight[1])
        self.mode = mode
        self.sparse = sparse

        self.embedding_layer_q = nn.Embedding(self.num_embeddings[0], self.embedding_dim[0])
        self.embedding_layer_r = nn.Embedding(self.num_embeddings[1], self.embedding_dim[1])

        self.embedding_layer_q.weight = self.weight_q
        self.embedding_layer_r.weight = self.weight_r

    def reset_parameters(self):
        nn.init.uniform_(self.weight_q, -1.0, 1.0)
        nn.init.uniform_(self.weight_r, -1.0, 1.0)

    def forward(self, features):

        #nonzero_count = torch.count_nonzero(features, dim=1).view(-1, 1)
        #print(nonzero_count)

        nonzero_count = (512 - (features == 0).sum(dim=1)).view(-1, 1)
        #print(nonzero_count)
        #assert 0 == 1

        input = (features % self.num_categories).long()
        input_q = (input / float(self.num_collisions)).long()
        input_r = (input % self.num_collisions).long()        

        embed_q = self.embedding_layer_q(input_q)
        embed_r = self.embedding_layer_r(input_r)
        #print(temp_embed_q.shape)
        #print(temp_embed_r.shape)

        embed_q = torch.sum(embed_q, dim=1).view(-1, self.embedding_dim[0])
        embed_r = torch.sum(embed_r, dim=1).view(-1, self.embedding_dim[1])
        #print(temp_embed_q)
        #print(temp_embed_r)

        '''
        embed_q = F.embedding_bag(
            input_q,
            self.weight_q,
            offsets=None,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            mode=self.mode,
            sparse=self.sparse,
        )
        embed_r = F.embedding_bag(
            input_r,
            self.weight_r,
            offsets=None,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            mode=self.mode,
            sparse=self.sparse,
        )

        print(embed_q)
        print(embed_r)

        #assert 0 == 1
        '''

        embed_q = torch.div(embed_q, nonzero_count)
        embed_r = torch.div(embed_r, nonzero_count)

        if self.operation == "cat":
            embed = torch.cat((embed_q, embed_r), dim=1)
        elif self.operation == "add":
            embed = embed_q + embed_r
        elif self.operation == "mult":
            embed = embed_q * embed_r
        else:
            raise RuntimeError(f"Not valid operation: {self.operation}!")

        return embed


class InteractionBase(nn.Module):
    def forward(self, batch):
        """
        :param batch: [batch_size, #embeddings, embedding_dim]
        """
        raise NotImplementedError()

class DotInteraction(InteractionBase):
    def __init__(self, embedding_num: int = 0, embedding_dim: int = 0):
        """
        Interactions are among outputs of all the embedding tables and bottom MLP, total number of
        #embeddings vectors with size embedding_dim. ``dot`` product interaction computes dot product
        between any 2 vectors. Output of interaction will have shape [batch_size, C_{#embeddings}^{2}].
        """

        super(DotInteraction, self).__init__()

        self._num_interaction_inputs = embedding_num
        self._embedding_dim = embedding_dim
        self._tril_indices = torch.tensor(
            [
                [i for i in range(self._num_interaction_inputs) for _ in range(i)],
                [j for i in range(self._num_interaction_inputs) for j in range(i)],
            ]
        )

    def forward(self, batch):
        """
        :param batch: [batch_size, #embeddings, embedding_dim]
        """

        interaction = torch.bmm(batch, torch.transpose(batch, 1, 2))
        interaction_flat = interaction[:, self._tril_indices[0], self._tril_indices[1]]
        return interaction_flat

class MLPLayer(nn.Module):
    def __init__(self, input_size, output_size, hidden_dropout_prob=0.1, direct=False):
        super(MLPLayer, self).__init__()

        self.dense = nn.Linear(input_size, output_size)
        self.direct = direct
        if not self.direct:
            self.activation = nn.Tanh()
            self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input):
        output_1 = self.dense(input)

        if self.direct:
            return output_1

        output_2 = self.activation(output_1)
        return self.dropout(output_2)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=-1, hidden_dropout_prob=0.1):
        super(MLP, self).__init__()

        if isinstance(hidden_size, int):
            assert num_layers > 0, f"Wrong num of layers: {num_layers}"

            self.layers = nn.ModuleList(
                modules=[
                    MLPLayer(
                        input_size if i == 0 else hidden_size, hidden_size, hidden_dropout_prob=hidden_dropout_prob
                    )
                    for i in range(num_layers)
                ]
            )
            self.projection = nn.Linear(hidden_size, output_size)
        else:
            self.layers = nn.ModuleList(
                modules=[
                    MLPLayer(
                        input_size if i == 0 else hidden_size[i - 1],
                        hidden_size[i],
                        hidden_dropout_prob=hidden_dropout_prob,
                    )
                    for i in range(len(hidden_size))
                ]
            )
            self.projection = nn.Linear(hidden_size[-1], output_size)

    def forward(self, input: Tensor) -> Tensor:
        hidden_states = input
        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states)
        output = self.projection(hidden_states)
        return output

class MEB(nn.Module):
    def __init__(self):
        super(MEB, self).__init__()
        embedding_q = nn.ModuleDict({
            "model": QREmbeddingBag(num_categories=cur_num_categories, num_collisions=cur_num_collisions, embedding_dim=128, padding_id=0, operation="cat", mode="sum")
        })
        embedding_r = nn.ModuleDict({
            "model": QREmbeddingBag(num_categories=cur_num_categories, num_collisions=cur_num_collisions, embedding_dim=128, padding_id=0, operation="cat", mode="sum")
        })
        interaction = DotInteraction(embedding_num=4, embedding_dim=128)
        meb_task = nn.ModuleDict({
            "head": MLP(input_size=6, hidden_size=[8, 4, 2], output_size=1)
        })
        
        self.nodes = nn.ModuleDict({
            "embedding_q": embedding_q,
            "embedding_r": embedding_r,
            "interaction": interaction,
            "meb_task": meb_task
        })

    def forward(self, input_q, input_r):
        res_embedding_q = self.nodes["embedding_q"]["model"](input_q)
        res_embedding_q_q = res_embedding_q[:, :128].unsqueeze(1)
        res_embedding_q_r = res_embedding_q[:, 128:].unsqueeze(1)

        res_embedding_r = self.nodes["embedding_r"]["model"](input_r)
        res_embedding_r_q = res_embedding_r[:, :128].unsqueeze(1)
        res_embedding_r_r = res_embedding_r[:, 128:].unsqueeze(1)

        res_embed = torch.cat((res_embedding_q_q, res_embedding_q_r, res_embedding_r_q, res_embedding_r_r), dim=1)
        #print(res_embed.shape)

        res_interaction = self.nodes["interaction"](res_embed)
        #print(res_interaction.shape)

        res_meb_task = self.nodes["meb_task"]["head"](res_interaction)
        #print(res_meb_task)
        res_score = torch.sigmoid(res_meb_task)
        return res_score

model = MEB()
model.eval()

cur_state_dict = torch.load("./meb.bin", map_location=torch.device("cpu"))

print(cur_state_dict.keys())
cur_state_dict["nodes.embedding_q.model.weight_q"][0] = torch.zeros(128)
cur_state_dict["nodes.embedding_q.model.weight_r"][0] = torch.zeros(128)
cur_state_dict["nodes.embedding_r.model.weight_q"][0] = torch.zeros(128)
cur_state_dict["nodes.embedding_r.model.weight_r"][0] = torch.zeros(128)

with torch.no_grad():
    for name, param in model.named_parameters():
        print(name)
        param.copy_(cur_state_dict[name])

'''
print(cur_state_dict["nodes.embedding_q.model.weight_q"])
print(cur_state_dict["nodes.embedding_q.model.weight_r"])
print(cur_state_dict["nodes.embedding_r.model.weight_q"])
print(cur_state_dict["nodes.embedding_r.model.weight_r"])
assert 0 == 1
'''

feature_str = "00FFCC9F4F6B2DA58339 01081A62CB7C78C61EBE 01706436649A9AAE6EAC 01718284C7C7350CD914 02507CB54270247ACDCC 0250966A878541CC54CE 025041FE31546C1EC48D 0251D23AC19E92B0CB60 025160BCDB6C7576DA23 0251FAA4A0EB3E6EFC8A 03506CC50D8769E667AE 0350B4A1E9597119CA3C 03519F8D3C1FFCADA8DF 0351AD6ED3BC5570F5F1 015064E3F2E07290C56D 015151B733F2EFC250CB 0151460FA839DBA05C62 015130B7EA17958BFBF5 01513B93EF0F6F3FD3A4 01517375D584166BFB77 0402061DC04B9DB9B3C8 0403570F3A644D02A43D 0404E399EC409DF9B779 040467AEA0EDBEB357A0 04046BA8B7C39BAE5E7C 0405B87A2C850B6BCE66 0406A88592E16EC1DDAA 041018853BB85364C888 04107FFF5E3E5997CB6A 0410472619F50E7BB03E 04108D4326CE37EA2C24 04103BB2307D31EEF963 0410D28C401384F628CA 0410A0F994C8C979AD83 04102C44D776AA7ACADD 0410F8982000A3F92DBC 0410313AE8C1391E1202 04107294A7A69081BF64 0410E8ECDDE6AEDA1A42 0410E778A727956A5B31 041082975328FF7BFAC2 04105116E33463564F2C 041092CF80C7E02A1522 0410C61A8BBDEF068235 0410ADE043ED32D72381 041023F3154C8A79F85F 0410408BF6F8DA344925 0410FEECD9F541DA2B95 042186B1687E4489ACFE 04221D0D91326A3E71E9 042237FB97BCCB8AE308 0422A903807D97D626C6 0422C848072F25D61AB7 04229DFD8CCC413779D2 0423F62DC54C15F07FBF 0424A6D00FA623089052 0424E93912D6BA851709 043026F012BE69AFFAD5 04307CF0377BB112DD3D 04329B40C0B717C4DAC2 0433CACE3B8BE5FA9407 0415B9B1187906403769 041571992843F76D9AE5 0415C9C3DE6E57E3216D 0415BA685A51FE74980B 04158638724FD4A7DDCB 04452C15B1144921EC00 044594F368AC3C773855 0445FA2D1DD49778CEB5 0445BC0CD0545FE45DF7 0450EC5C18BE7CA346FE 04516F244CFD9872DB14 04516E523FC591928DEE 04513EE1B5823CE50801 04516103666C91938478 0451CB687AFE03C03BA0 04522650A223F2B9C3B2 05023449D8B53AF0E8C9 05032058A165FBC68241 05047359E2A046B4E5F3 05049AE5FA2C0187A419 0504F8CC3AC8754059D8 050569BF9BBA8FF2EC9F 0506330463C5BBE9AF07 0510BD0713AECD590A72 0510B3A4060E7961B76B 0510DE6F0CE35D0733BC 05100E713EC14ED4D5FA 0510ED894C93FCA0457D 0510E619A72FCB658E79 0510E4B2E83A346B65A3 0510300072A17510D845 0510D6B3DDB304690CB6 0510E31426E10D4AB26E 0510DDE324E94A6A92C7 05104D4BA2C140EF4637 05107FFF938AFA022B4D 0510DF6DD4488644A000 0510B6A98DE93D3A0D7A 0510F7D0959D5FA1B4A6 051003EB713AA49DECE1 0510361DA8535A550DF9 0510929B7AEAF0CB0737 05109AAB0FF26E624F6C 0510D40E1948FC0FF8B8 05215FC36092901D3516 0522A414A0882109EDEC 05224A56689A8607D1DE 0522BC5D5D0656ECF255 052240836D411B087FE8 05223A5EBD835D049FC3 0530F0C5FDC87A1FB08B 05301D7BA052EE0F0538 053224DF405A48FC4B02 0533058A7A2919F920BB 0515FCD706E241AD5249 051515FE7277E3BBA8AD 051540EF6604C78EB2B5 0515BEE2B85DEC178275 05150BDC26ADD467433A 0602F8D8321025EC0525 07027101D39E6CC890D4"
feature_str_1 = "00FFCC9F4F6B2DA58339 0108E22A066451F924A9 01704D5BBB56F07EFDAA 0171439BA18F4CE882FF 025040DFB08F46C2F4F7 0250D180940C282EECB2 025073729D0E4DB2F818 0250E1CFF0BC7898CC46 0250E5DC5B8A36CE90F5 0250586592DCA20EEA0E 0251B92B7D98AA5531B2 0251141BB8F66C5F03DD 02518F51121D5A056F07 0251541A39889868D3A5 025168F38AD935BA5610 025165AC91BCC32DBA8C 0350CD0CED90DBCFBCE1 0350D97BAAF1B7C5EDE5 035040539E80883EC8E8 0350BFFEB872BF7F1A58 0350C4637F99DA970A57 0351670FC4637FFDA818 0351E73096F9B7E4CCAE 03515E6E72E17848259C 03511B26144DB5C41275 03514AB7D90C981FFED4 0150F0D3610B858F8D94 01508E531D0D5A9DCC0D 0150645252B27F2B4AEA 01504F24246D55C57DE9 0150B11A05C90B9BD16A 0150ABD1569AAF057171 01512EC4692615E2C9EA 01518D9556B8418EA7B2 01518BAE0B129DE8B634 0151C5DC84984121509B 01515B48382CB2864B4E 01511CC08C7B3E1B11A2 0402CCD377F7A541BA2A 0403570F3A644D02A43D 04040FA57420C16A4F31 040442AD42686866C357 0404655CAD556B4E5B49 04045F1D21BA1B41212A 040450303C469EB7E711 0404E50571619561CB4A 04058F94728D236FAD9A 040688528D9BE969DD9C 0410F7FD505EF4534BAA 0410C1FAB8BE86E83FB8 04109D0BF7464F5B44DE 0410F8BF1E3D5A87F56C 0410877CFFA054AEC5E6 0410666AAEF93B43E4B7 0410BA8D03DF817C7788 0410607B4518955C0442 041050A37F6F92BCB8AE 0410E6D239118BB70050 0410A08F7547B15BCD18 04103178F9BE0F92C4B1 041095F446DBE5BA6A0F 0410D8F50D44433ADD7C 04100F831A2CB452DCD4 041041895DB145DCD888 0410CA10C72AFD9167E0 041049BB0E6B84625E39 04104FA5FB6167D42B49 04103EBE7FF252C1CC31 0410DEF37224656DF510 04109ADA8CCFFB67163F 04108ED1764BF026303D 04100149F76CE8BA6A03 0410D0F19FFDD55DE883 0410AFB0C1C822420B6D 0410ABB889B2355961D9 0410F29409EED2C6EC8A 0410EF149AF3C31802AD 04100F6760C211F96918 0410ED95D79591DCBD0E 041021EA1EBB71DF1E7E 04104EEA8E128486E141 0410CBDB61B5BB6EF84D 041003F7301877F2EE1E 04102089B9181A7B6028 0410A42F396E586F4A55 0410F98ED65E34AE1CDE 0410B3FFF127B513DEA2 0410E79B1863AF79950A 041069DC9A0B14F78FAD 04102A5D7D616F4DFF6A 0410EEDCD02E6DB0AA80 0410839DA7FC0177AC91 04107C4AAE5537669713 0410B27C44E18A5DE8DF 04104E56780BD288E94A 0410D94C323EFFD17563 0410AEBAD9980F8DBBCA 0410A26A7F6ACB8276AF 0410C33D8F73260439C3 04109F3167DF66807CE4 04105A9F41D80363396D 0410C9BE178D0B7F8A1E 041074971B21F8CB565C 04104FDD63ED95A8BA99 04108D6C1340E0E64F6C 0410E2445E6314AEA3A6 0410EDC80FD84BA751D7 0410D3BEF3361BA0774F 0410098A83765F3403D4 0410A61FAC8C336032DB 0410FEFB0D8DD6F827A1 0410A33C1373F99AC199 0410D8FA19D1B092586A 041001D1F0C4632C6B57 0410E991B4B44D699FEA 04100C7C479048D9E128 04105A31D2D8D3E4065B 041051FE0178B32ADC69 04108451D532B3EBEEF7 04108768FFF60752EE18 0410FA3713FB0CACC6D1 0410755375AD3FAD399A 04107DF9E66B17DA9E52 04106E5037A74892E071 0410D442AF3D5400F4A4 0410203E19DA042026D8 0421F6ADC37ADDF628C4 0421B8A73EECE38D16BE 04219CCBF52D33622D51 042145DB66777F0B9B82 0422AFF5E4565FBD46F2 042295BDF50BBF85EE40 04224B696714A6774DFC 0422AD72622CC7933AC7 04227C74C2CA835148E3 0422DA3C4DB085B5CB42 04224CE6282E6F085B2F 0422FC2EC3F54D2772CE 042223578433DF4F1473 0422EB5F8D755FFCA4B2 0422FDCE8EE11ED2E396 0423124E4E80EF66EB1A 0424D5A8FCF15BE865EC 04243463DE5F5C0D424B 0424298110E7911A095C 04240F7B034EE47ACEC5 0424F6583CE597D59CC7 0424037FACB05CDB7CCC 0424952ADF0BB2FBEF7F 042468DF03FB201EF0A5 04306362FA0213425AE5 04302A2B47CBA14E034A 04323D90C09165994B01 04336DEA4B7263CA57F6 0415950E9662B6B8FA6C 04150C1D725A266B2C71 041503E4B16599BB411D 04156765CB6AA2CD0DF1 0415EE0A57CFE5B1C27C 0445D9ED05CD2D5F1822 044526728234ED787253 0445B9299D533806749B 0445113CB67C6CFA9F54 0445A2CA921B48807662 044507A3045C6853EE0D 0445DBD2AD6927C6548F 0450F954245119D967A6 045046CCF6BA5C7AE16F 0450BB3B465AE3BE31B2 04504D5ED228CE0C05FA 045191A06B1AFEDBE2B0 0451A77C9E19A0B5DF77 045125FE060BE4DB9320 04518FE05F6C31BD2B0F 045179F5AADD2A67555C 045100CBFB29326F5742 0451FFADA8609C461394 0451FCA822C0F66C7202 04516C63A4E824B0099E 045151A1C5F4C0A59DF9 04515F9D6BAF56328988 04524429967D855948D9 05026BE6774D0270248E 05032058A165FBC68241 0504E90D058314C477E8 0504B73EC16127B54824 0504C05F8F95E99C8925 0504C45F20248C8B40AC 0504B02EFE214CF671AB 05044C3929F322092E1E 05051977FBCE9F8AE577 0506414B1C24B5D36CF7 051064E38613909FC482 05102854095B94917107 05107D471131725A263E 0510E12D3FDE70FF7274 0510C2992EEF2BA13A7D 0510F1CF2B5685C31165 051037042FA602F39615 051084EE08AF7CE2B422 0510E29A917BCBC4A078 0510A93D25D229F894E0 0510FDEAE127FE2593D5 0510AA169C7E42C66F81 051047F6DA555324B3DC 0510541F8EE1FF0DFD5C 05106E55FB56BB565E3B 051052D3CAB18534FEDA 0510DC90485A7A958A17 0510A7BE7DBD2EE5B6FA 0510370197AE9449C8F9 051034F5E08323A68D8F 05104D088B689343662F 0510F77AA22459B121A7 051097FDE6A1C2111FB2 05102884CAC0B77708E4 0510612DD8064BCDFCCB 0510CA26DB4DD4C09AC9 05103652588D4FD8DA06 05109283683E96AB1337 05108507A5B9365BD685 051035E845C13CB9F2FC 0510D461307AB4CA10CB 0510121592801787F008 0510A13A48DB7C0E1F7F 05102E8A4624421A8FA0 05109D3074094242EF5C 05107F44A6AC22CC7186 0521638E0C7DF6348EB3 05213278E287F79ED22A 0521248643CB8B0B0B30 0521E8F90E1EBD41CE5C 0521206E77E28700BEB2 0521A3A653DFEF94D1BD 0522C1CE029440E21C36 05223CE67E3B3C53D3F6 05220692462F86763350 0522D95A057F13A2FC3A 052292B28E4CFE91BB8C 05224C13A97F045EAE18 0515F60A8B809F77E574 0515FFACB559573331D5 051582A07B36249B2442 06025BB06DEE4E6C3F77 07027C9DF9893E03776A"
feature_str_2 = "00FFCC9F4F6B2DA58339 01080C4D7EA49F3069D1 01700D836A6C6243DB0A 017129D458711518E308 0250D59630CC994DAE3E 0250E53D21E94FE01E4A 02514B30DBC5CA1AB9D9 0251BCBCC057DDCBAAEB 0350A35A1549F4A86FA8 0351B48492D62D5D40A4 01509D2C8E09DAEE346A 0151903F8254A85A2245 015191EA5CC1FD415CF4 0151886D9E25899E33DD 0151808BAC42097DC553 01517789AA69CEA9D9AC 0151D1F3BAF55B45D02F 0151CD1B62AEB9B36E0B 015132933EBD1AE66F26 0151DA05706E6101B491 015165EB26DFC31FBAD9 015124E793404FAC0CD0 0151047BEDB4BC697900 04025025D8E2FB19D664 0403570F3A644D02A43D 0404693D9561E4C9AAF0 040427BFFB841A50C54F 04059C98F04113F958F3 04067CA14540905BCB9B 0410633FA9F17BEE4776 0410AA9072E846BF2F84 04109492EFB2677C35A3 0410BBD4497ED658C4D9 0410EA8D75E6CD5D04D1 04106DDF67B48E49357A 04107D1177550ED2D869 0410ABFCC7D1F649468B 041096F1E99B0E7AACF9 0410E61753C5CCA85133 041070089B7677F1159D 04101BDCA237B3116253 0410907F03FC080C51E5 04106AA8CFE93DA236C6 042182F3F553DD468CAC 042200DB29E1E6694A65 04225881CB2E6AB69EEB 0422BC5E1D20CF449DC4 042200D8FC0C7B0205CB 0422837A320987C9784B 0422515FE7BCF62F39CD 04247503E26502C1DF93 04304F3E70E2AA42B892 0415A3F8A9810CC59FEB 0415CDACC27E65EC2FDF 0415AA1D9C50AB7D9B76 04153A34A2455C154A00 0445714031A08F0F24DF 0445ED132479BBEA8AF2 0445F6412B271BF24EBB 045005E7598E6F229B60 04514DAAE116745F2796 0451AD5F5758DA1971F3 04513B61E2CCBA6EEF80 0451C56C0B1FB6979CB3 0451ADE0662F112E3439 04517F85225B1C185075 0452E2998AAA57FCBAB3 05020C6904E73FA51A3C 05032058A165FBC68241 0504030465024504C3A0 05045DE3949FD8E8E74A 0505B3E0A496EA30E07D 05063643F8E16EB379E4 05105C7E737EE79D0989 05107708A0DD9C929C65 05106C919CD461FA22D9 05101A86E054788C9A79 0510255426B690B55BFA 05103BA54AE13929F91C 051003F7440C1992FBFD 051076C8F6E8C0ECB782 05102DC2400201B58837 0510A68181CBC520475F 051070B642293BA66B71 0510A079DBAA9716A51C 051022CF0E906A6611A8 0510E51F56665BE4D987 05104737CF033D467570 05100764B01A6055139E 0510CAFDAC87F60D4E1A 0510168115518E101127 0510BBEAF0D59D968C02 0510854F2244E62BD43F 05101DE3011EBA8CFC8A 05105DE0D0856FC964C7 05107A12908DADD7C22D 0510D6AF9097653B6C4A 0510AE493899E4FFF048 05103597888AD97A84B1 05217ADA7EE9B8253A92 0522A414A0882109EDEC 0522D32A8E7733CF1DC5 0522F2E3142CE59C6C0A 0522C70BF39B1DD65A41 0522E311A2E4A7AACD0F 0522C677CAEFEDE87362 05228AFCEC6C8429547C 0522458BF68AEB7283A3 0522B47B0FCBCF6C52F4 05229438A25227543B4F 0522811D4787602E421A 0522B38812DF418444D8 0530E5A689EC643F60B3 05159EEA5940F77C2898 0515EBC124FF73FD4CC1 05157D1363ADB6BFECB7 05151B2C5C4524F79AFE 0602A527E78A6D57B5E7 07022ECD18B3E1C5F0FF"

splitted_feature = feature_str.split()[:cur_max_seq_length]
sparse_features = [int(s[4:], 16) % cur_maxsize for s in splitted_feature] + [cur_padding_id] * (cur_max_seq_length - len(splitted_feature))

splitted_feature_1 = feature_str_1.split()[:cur_max_seq_length]
sparse_features_1 = [int(s[4:], 16) % cur_maxsize for s in splitted_feature_1] + [cur_padding_id] * (cur_max_seq_length - len(splitted_feature_1))

splitted_feature_2 = feature_str_2.split()[:cur_max_seq_length]
sparse_features_2 = [int(s[4:], 16) % cur_maxsize for s in splitted_feature_2] + [cur_padding_id] * (cur_max_seq_length - len(splitted_feature_2))

cur_q = [[(item // math.floor(math.sqrt(sys.maxsize))) for item in sparse_features], [(item // math.floor(math.sqrt(sys.maxsize))) for item in sparse_features_1], [(item // math.floor(math.sqrt(sys.maxsize))) for item in sparse_features_2]]
#cur_q = [[(item // math.floor(math.sqrt(sys.maxsize))) for item in sparse_features], [(item // math.floor(math.sqrt(sys.maxsize))) for item in sparse_features_1]]
cur_q = torch.tensor(cur_q).long()

cur_r = [[item % math.floor(math.sqrt(sys.maxsize)) for item in sparse_features], [item % math.floor(math.sqrt(sys.maxsize)) for item in sparse_features_1], [item % math.floor(math.sqrt(sys.maxsize)) for item in sparse_features_2]]
#cur_r = [[item % math.floor(math.sqrt(sys.maxsize)) for item in sparse_features], [item % math.floor(math.sqrt(sys.maxsize)) for item in sparse_features_1]]
cur_r = torch.tensor(cur_r).long()

with torch.no_grad():
    print("cur_q: ", cur_q.size())
    print("cur_r: ", cur_r.size())
    res_score = model(cur_q, cur_r)
    print(res_score)

    #assert 0 == 1

    torch.onnx.export(model, 
                    (cur_q, cur_r), 
                    "meb_batch.onnx", 
                    export_params=True, 
                    opset_version=12, 
                    operator_export_type=OperatorExportTypes.ONNX,
                    do_constant_folding=True,
                    input_names = ["input_q", "input_r"],
                    output_names = ["score"],
                    dynamic_axes = {'input_q' : {0 : 'batch_size'},
                                    'input_r' : {0 : 'batch_size'}, 
                                    'score' : {0 : 'batch_size'}
                                    })