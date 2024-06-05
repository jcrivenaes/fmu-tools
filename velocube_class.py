from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
import xtgeo
from scipy.interpolate import griddata, interpn

SURFS = ["MSL", "TopVolantis", "BaseVolantis", "BaseVelmodel"]

TEMPLATE_CUBE = "seismic--amplitude_far_time--20180101"

TIME_CUBE = TEMPLATE_CUBE

DCAT = "DS_velmod"

TCAT = "TS_time_extracted"

ZMIN = 1200
ZMAX = 2400
ZINC = 2

VELO_LIMITS = (1480, 5000)

PRJ = project  # type: ignore # noqa # pylint: disable=undefined-variable


@dataclass
class VelocityCube:
    cube_template: xtgeo.Cube  # shall be in TWT?
    depth_surfaces: list
    time_surfaces: list
    use_interval_velocity: bool = False

    vcube_t: xtgeo.Cube = field(default=None, init=False)  # velocity cube in time
    surf: xtgeo.RegularSurface = field(default=None, init=False)
    velo_surfaces: list = field(default=False, init=False)

    def __post_init__(self):
        self._derive_templates()
        self.depth_surfaces = self._resample_surfaces(
            self.depth_surfaces,
            self.cube_template,
            fill=True,
            ensure_consistency=True,
        )
        self.time_surfaces = self._resample_surfaces(
            self.time_surfaces,
            self.cube_template,
            fill=True,
            ensure_consistency=True,
        )
        if self.use_interval_velocity:
            self._velo_maps_interval()
            self._create_velo_cubes_interval()
            self._create_velo_cube_average_from_interval()
        else:
            self._velo_maps_average()
            self._create_velo_cubes_average()

    def _derive_templates(self):
        # find min and max of depth and time surfaces
        dmin = self.depth_surfaces[0].values.min()
        dmax = self.depth_surfaces[-1].values.max()
        tmin = self.time_surfaces[0].values.min()
        tmax = self.time_surfaces[-1].values.max()

        print(dmin, dmax, tmin, tmax)

        self.dcube = self.cube_template.copy()
        self.dcube.values = 0.0
        self.vcube_t = self.dcube.copy()

    @staticmethod
    def _resample_surfaces(surflist, cube, fill=False, ensure_consistency=False):
        def _ensure_consistency(slist):
            """Ensure consistensy for depth or time surfaces"""
            for inum in range(1, len(slist)):
                values0 = slist[inum - 1]
                slist[inum].values = np.where(
                    slist[inum].values < values0.values,
                    values0.values,
                    slist[inum].values,
                )

            return slist

        tmpl = xtgeo.surface_from_cube(cube, cube.zori)
        new_surfs = []
        for surf in surflist:
            tmpl.resample(surf)
            tmpl.fill()
            new_surfs.append(tmpl.copy())

        if ensure_consistency:
            return _ensure_consistency(new_surfs)

        return new_surfs

    def _velo_maps_interval(self):
        """Create velocity maps per interval"""

        vel = []
        for no in range(1, len(self.depth_surfaces)):
            t0 = self.time_surfaces[no - 1]
            t1 = self.time_surfaces[no]
            d0 = self.depth_surfaces[no - 1]
            d1 = self.depth_surfaces[no]

            vspeed = d1.copy()
            tdiff = t1.values - t0.values
            tdiff[tdiff < 0.1] = 0.1
            vspeed.values = (d1.values - d0.values) / tdiff
            vspeed.values *= 2000
            stddev = np.ma.std(vspeed.values)
            mean = np.ma.mean(vspeed.values)
            low = mean - 2 * stddev
            hig = mean + 2 * stddev

            vspeed.values[vspeed.values < low] = low
            vspeed.values[vspeed.values > hig] = hig
            vel.append(vspeed)

        self.velo_surfaces = vel

    def _create_velo_cubes_interval(self):
        """Create velocity cubes (in Depth/Time) as interval"""

        tc = self.vcube_t
        vcube = tc.values.copy()
        darr = [tc.zori + n * tc.zinc for n in range(vcube.shape[2])]
        print(darr)
        vcube[:, :, :] = darr

        for num, _ in enumerate(self.velo_surfaces):
            vmap = self.velo_surfaces[num].copy()
            surf = self.time_surfaces[num].copy()
            tval = np.expand_dims(surf.values, axis=2)
            vval = np.expand_dims(vmap.values, axis=2)
            self.vcube_t.values = np.where(vcube >= tval, vval, self.vcube_t.values)

    def _velo_maps_average(self):
        """Create average velocities from MSL to surface N"""

        vel = []
        for no in range(1, len(self.depth_surfaces)):
            t0 = self.time_surfaces[0]
            t1 = self.time_surfaces[no]
            d0 = self.depth_surfaces[0]
            d1 = self.depth_surfaces[no]

            vspeed = d1.copy()
            tdiff = t1.values - t0.values
            vspeed.values = np.divide((d1.values - d0.values), tdiff)
            vspeed.values *= 2000
            vel.append(vspeed)

        vel.insert(0, vel[0])
        self.velo_surfaces = vel

    def _create_velo_cubes_average(self):
        """Create velocity cubes (in Depth/Time) as average"""

        tc = self.vcube_t
        tcube = tc.values.copy()
        time_arr = [tc.zori + n * tc.zinc for n in range(tcube.shape[2])]

        vcube = np.zeros_like(tcube)
        tlen = len(self.time_surfaces)
        vlen = len(self.velo_surfaces)

        assert vlen == tlen

        for i in range(tcube.shape[0]):
            for j in range(tcube.shape[1]):
                tmap = [self.time_surfaces[num].values[i, j] for num in range(tlen)]
                vmap = [self.velo_surfaces[num].values[i, j] for num in range(vlen)]

                vcube[i, j, :] = np.interp(time_arr, tmap, vmap)

        self.vcube_t.values = vcube

    # def _create_velo_cube_average(self):
    #     """Create average velocity cube"""

    #     vt = self.vcube_t.copy()
    #     darr = [vt.zori + n * vt.zinc for n in range(vt.values.shape[2])]
    #     vt.values[:, :, :] = darr

    #     ilines, xlines, tlines = self.vcube_t.dimensions

    #     tlen = len(self.time_surfaces)

    #     print("Make velo cube...")
    #     for il in range(ilines):
    #         for xl in range(xlines):
    #             tstack = vcube_t_z.values[il, xl, :]
    #             tmap = [self.time_surfaces[num].values[il, xl] for num in range(tlen)]
    #             vmap = [self.velo_surfaces[num].values[il, xl] for num in range(tlen)]
    #             self.vcube_t.values[il, xl, :] = np.interp(tstack, tmap, vmap)

    #     print("Make velo cube... DONE!")

    def _create_velo_cube_average_from_interval(self):
        """Create average velocity cube from interval velocity cube."""

        # Compute rolling mean of the input array
        ni, nj, nk = self.vcube_t.dimensions

        arr = self.vcube_t.values
        cum = np.cumsum(arr, axis=2)
        cnt = np.arange(1, arr.shape[2] + 1)
        cnt = cnt.reshape(1, 1, -1)
        avg = cum / cnt

        self.vcube_t.values = avg

    def depth_convert_cube(self, incube: xtgeo.Cube, zinc=1, maxdepth=2000):
        """Use the current average velocity model/cube to perform depth conversion."""

        if incube.zori != 0:
            raise ValueError("The input cube must start at TWT = 0.0")

        seismic_attribute_cube = incube.values
        velocity_cube = self.vcube_t.values

        dt = incube.zinc / 2000  # TWT in ms. to one way time in s.
        times = incube.copy()
        tarr = [incube.zori + n * dt for n in range(incube.values.shape[2])]
        times.values[:, :, :] = tarr
        depth_cube = velocity_cube * times.values

        new_nlay = int(maxdepth / zinc)

        dcube = xtgeo.Cube(
            xori=incube.xori,
            yori=incube.yori,
            zori=0.0,
            ncol=incube.ncol,
            nrow=incube.nrow,
            nlay=new_nlay,
            xinc=incube.xinc,
            yinc=incube.yinc,
            zinc=zinc,
            rotation=incube.rotation,
            yflip=incube.yflip,
            values=0.0,
        )
        zmax = dcube.nlay * dcube.zinc

        new_depth_axis = np.arange(0, zmax, zinc).astype("float")
        seismic_attribute_depth_cube = dcube.values

        # Perform the interpolation for each (x, y) location
        for i in range(seismic_attribute_cube.shape[0]):
            for j in range(seismic_attribute_cube.shape[1]):
                depth_trace = depth_cube[i, j, :]
                seismic_trace = seismic_attribute_cube[i, j, :]

                seismic_attribute_depth_cube[i, j, :] = np.interp(
                    new_depth_axis,
                    depth_trace,
                    seismic_trace,
                    left=np.nan,
                    right=np.nan,
                )

        dcube.values = seismic_attribute_depth_cube
        return dcube

    def depth_convert_surfaces(self, insurfs: List[xtgeo.RegularSurface]):
        """Use the current average velocity model/cube to perform depth conversion."""
        print("Depth convert surfaces...")
        vcube = self.vcube_t  # velocity cube in TWT
        # tarr = [vcube.zori + n * vcube.zinc for n in range(vcube.values.shape[2])]
        tarr = vcube.zori + np.arange(vcube.values.shape[2]) * vcube.zinc  # TWT array
        original_surf = insurfs[0].copy()
        insurfs = self._resample_surfaces(insurfs, vcube)

        nx, ny = insurfs[0].dimensions

        for surf in insurfs:
            # Extract TWT values and mask
            twt_values = surf.values.data
            mask = surf.values.mask

            # Vectorized interpolation
            depths = np.zeros_like(twt_values)
            valid_mask = ~mask

            # Use advanced indexing for valid TWT values
            twt_valid = twt_values[valid_mask].flatten()
            i_indices, j_indices = np.nonzero(valid_mask)

            # Interpolate velocities
            velocities = np.array(
                [
                    np.interp(t, tarr, vcube.values[i, j, :])
                    for t, i, j in zip(twt_valid, i_indices, j_indices)
                ]
            )

            # Calculate depth
            depths[valid_mask] = twt_valid * velocities / 2000

            # Update surface values with depth-converted values
            surf.values = np.ma.array(depths, mask=mask)

        result = []
        for srf in insurfs:
            smp = original_surf.copy()
            smp.resample(srf)
            result.append(smp)

        print("Depth convert surfaces... DONE")
        return result

    def depth_convert_surfaces_old(self, insurfs: List[xtgeo.RegularSurface]):
        """Use the current average velocity model/cube to perform depth conversion."""
        print("Depth convert surfaces...")
        vcube = self.vcube_t  # velocity cube in TWT
        tarr = [vcube.zori + n * vcube.zinc for n in range(vcube.values.shape[2])]

        original_surf = insurfs[0].copy()
        insurfs = self._resample_surfaces(insurfs, vcube)
        print(insurfs)

        nx, ny = insurfs[0].dimensions

        for surf in insurfs:
            depth_surface = np.zeros_like(surf.values)
            for i in range(nx):
                for j in range(ny):
                    if surf.values.mask[i, j]:
                        continue
                    twt = surf.values[i, j]
                    varr = vcube.values[i, j, :]

                    # interpolate to find the exact velocity
                    velo = np.interp([twt], tarr, varr)
                    depth = velo[0] * twt / 2000

                    depth_surface[i, j] = depth

            surf.values = depth_surface

        result = []
        for srf in enumerate(insurfs):
            smp = original_surf.copy()
            smp.resample(srf)
            result.append(smp)

        print("Depth convert surfaces... DONE")
        return result

    # def _dconvert_cube_interval(self, incube: xtgeo.Cube):
    #     """Perform depth conversion using interval velocity cube."""

    #     data3d = incube.values
    #     dt = incube.zinc / 2000  # Time sampling interval in seconds
    #     depth = 0  # Constant depth of the seismic data in meters

    #     vp = self.vcube_t.values

    #     # Calculate the time axis
    #     tmax = dt * data3d.shape[2]
    #     time = np.linspace(dt, tmax, data3d.shape[2])

    #     # Calculate the depth axis using 3D velocity data
    #     depth_data = np.zeros_like(vp)
    #     depth_data[:, :, 0] = depth
    #     depth_data[:, :, 1] = np.cumsum(
    #         vp[:, :, :-1] * dt / 2 + vp[:, :, 1:] * dt / 2, axis=2
    #     )

    #     # Allocate memory for depth-converted seismic data
    #     data3d_depth = np.zeros_like(data3d, dtype=np.float32)

    #     # Perform depth conversion of the seismic data
    #     for i in range(data3d.shape[0]):
    #         for j in range(data3d.shape[1]):
    #             data3d_depth[i, j, :] = np.interp(
    #                 depth_data[i, j, :], depth_data[i, j, :] + depth, data3d[i, j, :]
    #             )

    #     newcube = incube.copy()
    #     newcube.values = data3d_depth
    #     return newcube

    # def _dconvert_cube_average(self, incube: xtgeo.Cube, z_inc: float = 2):
    #     """Perform depth conversion using average velocity cube."""
    #     reference_depth = 0.0
    #     velocity_cube_values = self.vcube_t.values

    #     cumulative_time_cube = np.cumsum(incube, axis=2)
    #     cumulative_distance = (
    #         2000 * incube.zinc * np.cumsum(velocity_cube_values, axis=2)
    #     )

    #     cumulative_time = cumulative_time_cube[:, :, np.newaxis]
    #     depth_cube_values = (
    #         cumulative_distance
    #         + reference_depth
    #         + cumulative_time * velocity_cube_values
    #     )

    #     depth_cube_values = 2000 * incube.zinc * np.cumsum(self.vcube_t, axis=2)
    #     depth_samples = np.arange(self.vcube_t.nlay) * z_inc

    #     newvalues = np.zeros(self.vcube_t.dimensions, dtype=np.float32)

    #     ilines, xlines, layers = incube.dimensions

    #     for il in range(ilines):
    #         for xl in range(xlines):
    #             newvalues[il, xl, :] = np.interp(
    #                 depth_samples,
    #                 depth_cube_values[il, xl, :],
    #                 incube.values[il, xl, :],
    #             )

    #     result = self.vcube_t.copy()
    #     result.values = newvalues
    #     return result


def main():
    tmplcube = xtgeo.cube_from_roxar(PRJ, TEMPLATE_CUBE)
    dlist = []
    tlist = []
    for sname in SURFS:
        dlist.append(xtgeo.surface_from_roxar(PRJ, sname, DCAT))
        tlist.append(xtgeo.surface_from_roxar(PRJ, sname, TCAT))

    velomod = VelocityCube(
        cube_template=tmplcube,
        depth_surfaces=dlist,
        time_surfaces=tlist,
        use_interval_velocity=False,
    )
    velomod.vcube_t.to_roxar(PRJ, "vcube_in_time_avg")

    # timecube = xtgeo.cube_from_roxar(PRJ, TIME_CUBE)
    # new_depth_cube = velomod.depth_convert_cube(timecube)
    # new_depth_cube.to_roxar(PRJ, "jriv_dconvert_average")

    # depth_convert depth surface
    new_dlist = velomod.depth_convert_surfaces(tlist)

    print("Store in RMS...")
    for n, s in enumerate(new_dlist):
        s.to_roxar(PRJ, f"surf_{n}", "", stype="clipboard")


if __name__ == "__main__":
    main()
