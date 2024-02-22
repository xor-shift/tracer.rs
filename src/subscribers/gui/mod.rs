use crate::subscriber::Subscriber;

pub struct GUISubscriber {}

impl GUISubscriber {
    pub fn new() -> GUISubscriber {
        Self {}
    }
}

impl Subscriber for GUISubscriber {

}
