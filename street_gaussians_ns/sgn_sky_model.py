import torch
import nvdiffrast.torch as dr
import torch.nn.functional as F
import kornia


class EnvLight(torch.nn.Module):

    def __init__(self, resolution=1024):
        super().__init__()
        self.base = torch.nn.Parameter(
            0.5 * torch.ones(6, resolution, resolution, 3, requires_grad=True),
        )

    def get_world_directions(self, camera, train=False):
        W, H = int(camera.width.item()), int(camera.height.item())
        cx = camera.cx.item()
        cy = camera.cy.item()
        fx = camera.fx.item()
        fy = camera.fy.item()
        c2w = camera.camera_to_worlds[0]
        grid = kornia.utils.create_meshgrid(
            H, W, normalized_coordinates=False, device="cuda"
        )[0]
        u, v = grid.unbind(-1)
        if train:
            directions = torch.stack(
                [
                    (u - cx + torch.rand_like(u)) / fx,
                    (v - cy + torch.rand_like(v)) / fy,
                    torch.ones_like(u),
                ],
                dim=0,
            )
        else:
            directions = torch.stack(
                [(u - cx + 0.5) / fx, (v - cy + 0.5) / fy, torch.ones_like(u)], dim=0
            )
        directions = F.normalize(directions, dim=0)
        R_edit = torch.diag(
            torch.tensor([1, -1, -1], device=c2w.device, dtype=c2w.dtype)
        )
        directions = (c2w[:3, :3] @ R_edit @ directions.reshape(3, -1)).reshape(3, H, W)
        return directions

    def forward(self, camera, train=False):
        dir_each_uv = self.get_world_directions(camera, train).permute(1, 2, 0)
        dir_each_uv = dir_each_uv.contiguous()
        prefix = dir_each_uv.shape[:-1]
        if len(prefix) != 3:  # reshape to [B, H, W, -1]
            dir_each_uv = dir_each_uv.reshape(1, 1, -1, dir_each_uv.shape[-1])

        light = dr.texture(
            self.base[None, ...],
            dir_each_uv,
            filter_mode="linear",
            boundary_mode="cube",
        )
        light = light.view(*prefix, -1)

        return light
