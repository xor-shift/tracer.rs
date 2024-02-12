use stuff::FloatingPoint;

use crate::types::*;

#[derive(Clone, Copy, PartialEq)]
pub struct LinearRGB(pub Vec3);

#[derive(Clone, Copy, PartialEq)]
pub struct SRGB(pub Vec3);

#[derive(Clone, Copy, PartialEq)]
pub struct XYZ(pub Vec3);

impl SRGB {
    fn srgb_oetf<T: FloatingPoint>(linear: T) -> T {
        if linear <= T::as_from(0.00031308) {
            T::as_from(12.92) * linear
        } else {
            T::as_from(1.055) * linear.powf(T::one() / T::as_from(2.4)) - T::as_from(0.055)
        }
    }

    #[allow(dead_code)]
    fn srgb_oetf_vec(linear: Vec3) -> Vec3 {
        let above_threshold = Vec3::gt(&linear, &Vec3::new_explode(0.00031308));

        let below = linear * 12.92;
        let above = linear.powf_scalar(1. / 2.4) * 1.055 - Vec3::new_explode(0.055);

        below * !above_threshold + above * above_threshold
    }

    fn srgb_eotf<T: FloatingPoint>(srgb: T) -> T {
        if srgb <= T::as_from(0.04045) {
            srgb / T::as_from(12.92)
        } else {
            ((srgb + T::as_from(0.055)) / T::as_from(1.055)).powf(T::as_from(2.4))
        }
    }

    #[allow(dead_code)]
    fn srgb_eotf_vec(srgb: Vec3) -> Vec3 {
        let above_threshold = Vec3::gt(&srgb, &Vec3::new_explode(0.04045));

        let below = srgb / 12.92;
        let above = ((srgb + Vec3::new_explode(0.055)) / 1.055).powf_scalar(2.4);

        below * !above_threshold + above * above_threshold
    }
}

impl From<XYZ> for SRGB {
    fn from(v: XYZ) -> SRGB { Into::<LinearRGB>::into(v).into() }
}

impl From<LinearRGB> for SRGB {
    fn from(v: LinearRGB) -> SRGB {
        SRGB(stuff::smallvec::Vector::<Float, 3>([
            SRGB::srgb_oetf(v.0[0]), //
            SRGB::srgb_oetf(v.0[1]),
            SRGB::srgb_oetf(v.0[2]),
        ]))
    }
}

impl From<LinearRGB> for XYZ {
    fn from(v: LinearRGB) -> XYZ {
        let matrix = stuff::smallvec::Matrix::<Float, 3, 3>([
            0.4124, 0.3576, 0.1805, //
            0.2125, 0.7152, 0.0722, //
            0.0193, 0.1192, 0.9505, //
        ]);

        XYZ(matrix * v.0)
    }
}

impl From<SRGB> for XYZ {
    fn from(v: SRGB) -> XYZ { Into::<LinearRGB>::into(v).into() }
}

impl From<XYZ> for LinearRGB {
    fn from(v: XYZ) -> LinearRGB {
        let matrix = stuff::smallvec::Matrix::<Float, 3, 3>([
            3.2406255, -1.5372080, -0.4986285, //
            -0.9689307, 1.8757561, 0.0415175, //
            0.0557101, -0.2040211, 1.0569959, //
        ]);

        LinearRGB(matrix * v.0)
    }
}

impl From<SRGB> for LinearRGB {
    fn from(v: SRGB) -> LinearRGB {
        LinearRGB(stuff::smallvec::Vector::<Float, 3>([
            SRGB::srgb_eotf(v.0[0]), //
            SRGB::srgb_eotf(v.0[1]),
            SRGB::srgb_eotf(v.0[2]),
        ]))
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct XYY(pub Vec3);

impl From<XYZ> for XYY {
    fn from(v: XYZ) -> XYY {
        let sum = v.0[0] + v.0[1] + v.0[2];

        XYY(Vec3::new([v.0[0] / sum, v.0[1] / sum, v.0[1]]))
    }
}

impl From<XYY> for XYZ {
    fn from(v: XYY) -> XYZ {
        let [x, y, big_y] = v.0 .0;
        let z = 1. - x - y;

        XYZ(Vec3::new([big_y * x / y, big_y, big_y * z / y]))
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    fn vectors_are_close(x: Vec3, y: Vec3) -> bool {
        let v = (x - y).abs();
        v.0.iter().all(|v| *v < 0.01)
    }

    #[test]
    fn test_select_values() {
        assert!(vectors_are_close(color::XYZ::from(color::SRGB(Vec3::new_explode(0.))).0, Vec3::new_explode(0.)));
        assert!(vectors_are_close(
            color::XYZ::from(color::SRGB(Vec3::new_explode(10. / 255.))).0,
            Vec3::new([0.0028850239786317, 0.003035269835488375, 0.0033054088508468404])
        ));
    }

    #[test]
    fn test_roundtrip_srgb_xyz() {
        for i in 0..=255 {
            for j in 0..=255 {
                let srgb = color::SRGB(Vec3::new([i as Float / 255., i as Float / 255., j as Float / 255.]));
                let xyz = color::XYZ::from(srgb);
                let srgb_again = color::SRGB::from(xyz);

                assert!(vectors_are_close(srgb.0, srgb_again.0));
            }
        }
    }
}
