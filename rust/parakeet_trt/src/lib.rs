#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use parakeet_trt_sys::*;
use std::ffi::{CStr, CString};
use std::ptr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParakeetError {
    #[error("Failed to create session")]
    SessionCreationFailed,
    #[error("Inference error: {0}")]
    InferenceError(String),
}

pub enum TranscriptionEvent {
    PartialText { segment_id: i32, text: String },
    FinalText { segment_id: i32, text: String },
    Error { message: String },
}

pub struct ParakeetSessionSafe {
    inner: *mut ParakeetSession,
}

impl ParakeetSessionSafe {
    pub fn new(model_dir: &str, device_id: i32, use_fp16: bool) -> Result<Self, ParakeetError> {
        let c_model_dir = CString::new(model_dir).unwrap();
        let config = ParakeetConfig {
            model_dir: c_model_dir.as_ptr(),
            device_id,
            use_fp16,
        };

        let inner = unsafe { parakeet_create_session(&config) };
        if inner.is_null() {
            return Err(ParakeetError::SessionCreationFailed);
        }

        Ok(Self { inner })
    }

    pub fn reset(&self) {
        unsafe { parakeet_reset_utterance(self.inner) };
    }

    pub fn push_features(&self, features: &[f32], num_frames: usize) -> Result<(), ParakeetError> {
        let res = unsafe { parakeet_push_features(self.inner, features.as_ptr(), num_frames) };
        if res < 0 {
            return Err(ParakeetError::InferenceError(format!("Error code {}", res)));
        }
        Ok(())
    }

    pub fn poll_event(&self) -> Option<TranscriptionEvent> {
        let mut event = ParakeetEvent {
            type_: 0,
            segment_id: 0,
            text: ptr::null(),
            error_message: ptr::null(),
        };

        let polled = unsafe { parakeet_poll_event(self.inner, &mut event) };
        if !polled {
            return None;
        }

        match event.type_ {
            ParakeetEventType_PARAKEET_EVENT_PARTIAL_TEXT => Some(TranscriptionEvent::PartialText {
                segment_id: event.segment_id,
                text: unsafe { CStr::from_ptr(event.text) }.to_string_lossy().into_owned(),
            }),
            ParakeetEventType_PARAKEET_EVENT_FINAL_TEXT => Some(TranscriptionEvent::FinalText {
                segment_id: event.segment_id,
                text: unsafe { CStr::from_ptr(event.text) }.to_string_lossy().into_owned(),
            }),
            ParakeetEventType_PARAKEET_EVENT_ERROR => Some(TranscriptionEvent::Error {
                message: unsafe { CStr::from_ptr(event.error_message) }
                    .to_string_lossy()
                    .into_owned(),
            }),
            _ => None,
        }
    }
}

impl Drop for ParakeetSessionSafe {
    fn drop(&mut self) {
        unsafe { parakeet_destroy_session(self.inner) };
    }
}

unsafe impl Send for ParakeetSessionSafe {}
unsafe impl Sync for ParakeetSessionSafe {}
