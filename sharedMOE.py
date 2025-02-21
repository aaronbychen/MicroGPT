class ShareExpertMOE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.moe_model = SparseMOE(config)
        self.shared_experts = nn.ModuleList(
            [
                BasicExpert(
                    config.hidden_dim, config.hidden_dim
                ) for _ in range(config.shared_experts_number)
            ]
        )

    def forward(self, x):
        sparse_moe_out, router_logits = self.moe_model(x)
        
        shared_experts_out = [
            expert(x) for expert in self.shared_experts
        ]
        
        shared_experts_out = torch.stack(
            shared_experts_out, dim=0
        ).sum(dim=0)
        
        return sparse_moe_out + shared_experts_out, router_logits


def test_share_expert_moe():
    x = torch.rand(2, 4, 16)
    config = MOEConfig(16, 2, 2)
    share_expert_moe = ShareExpertMOE(config)
    out = share_expert_moe(x)
    print(out[0].shape, out[1].shape)

test_share_expert_moe()