
#[derive(Clone, Copy)]
pub struct AABBox<S> {
    pub min: cgmath::Point3<S>,
    pub max: cgmath::Point3<S>,
}

impl<S: std::cmp::PartialOrd + Copy> AABBox<S> {
    pub fn extend(&self, other: AABBox<S>) -> AABBox<S> {
        Self {
            min: cgmath::point3(
                if self.min.x < other.min.x { self.min.x } else { other.min.x },
                if self.min.y < other.min.y { self.min.y } else { other.min.y },
                if self.min.z < other.min.z { self.min.z } else { other.min.z },
            ),
            max: cgmath::point3(
                if self.min.x < other.min.x { other.min.x } else { self.min.x },
                if self.min.y < other.min.y { other.min.y } else { self.min.y },
                if self.min.z < other.min.z { other.min.z } else { self.min.z },
            ),
        }
    }

    
}
