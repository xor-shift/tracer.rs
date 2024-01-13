use super::*;

#[allow(dead_code)]
fn visualise_spectrum() {
    let mut image = stuff::qoi::Image::new(256, 64);

    for i in 0..image.width() {
        let spectrum_start = 380.;
        let spectrum_end = 780.;
        let spectrum_cur = (i as Float / image.width() as Float) * (spectrum_end - spectrum_start) + spectrum_start;

        let xyz_analytic = color::XYZ(Vec3::new(color::cie_color_match::xyz_analytic(spectrum_cur)));
        let xyz_lookup = color::XYZ(Vec3::new(color::cie_color_match::xyz_lookup(spectrum_cur)));
        let srgb_analytic = color::SRGB::from(color::LinearRGB(color::LinearRGB::from(xyz_analytic).0.clamp(0., 1.)));
        let srgb_lookup = color::SRGB::from(color::LinearRGB(color::LinearRGB::from(xyz_lookup).0.clamp(0., 1.)));

        for j in 0..image.height() / 2 {
            *image.pixel_mut(i as usize, j as usize) = stuff::qoi::Color {
                r: (srgb_analytic.0[0].clamp(0., 1.) * 255.).trunc() as u8,
                g: (srgb_analytic.0[1].clamp(0., 1.) * 255.).trunc() as u8,
                b: (srgb_analytic.0[2].clamp(0., 1.) * 255.).trunc() as u8,
                a: 255,
            };
        }

        for j in image.height() / 2..image.height() {
            *image.pixel_mut(i as usize, j as usize) = stuff::qoi::Color {
                r: (srgb_lookup.0[0].clamp(0., 1.) * 255.).trunc() as u8,
                g: (srgb_lookup.0[1].clamp(0., 1.) * 255.).trunc() as u8,
                b: (srgb_lookup.0[2].clamp(0., 1.) * 255.).trunc() as u8,
                a: 255,
            };
        }
    }

    let mut file = std::fs::File::create("spectrum.qoi").unwrap();
    image.encode_to_writer(&mut file).unwrap();
}

#[allow(dead_code)]
fn visualise_xy() {
    let mut image = stuff::qoi::Image::new(512, 512);

    for row in 0..image.height() {
        for col in 0..image.width() {
            let x = col as Float / image.width() as Float;
            let y = (image.height() - row - 1) as Float / image.height() as Float;
            let xyy = color::XYY(Vec3::new([x, y, 1.]));

            let xyz = color::XYZ::from(xyy);
            let srgb = color::SRGB::from(xyz);

            *image.pixel_mut(col as usize, row as usize) = stuff::qoi::Color {
                r: (srgb.0[0].clamp(0., 1.) * 255.).trunc() as u8,
                g: (srgb.0[1].clamp(0., 1.) * 255.).trunc() as u8,
                b: (srgb.0[2].clamp(0., 1.) * 255.).trunc() as u8,
                a: 255,
            };
        }
    }

    let mut file = std::fs::File::create("xy.qoi").unwrap();
    image.encode_to_writer(&mut file).unwrap();
}
