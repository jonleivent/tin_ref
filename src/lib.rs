//! TinRefs
//!
//! Tokenized "INterior" mutability REFerences, or
//! Tokenized INherited mutability REFerences
//!
//! Motivation: Provide a convenient (albiet a bit syntactically sour)
//! and no-runtime-overhead mechanism for arbitrary interior
//! mutability that is Copy, Sync and Send, and yet obeys all of
//! Rust's inherited mutability rules, while not requiring users to
//! make unsafe calls.
//!
//! How this works: An invariant lifetime parameter 'i (signifying
//! "identity" and "invariant") ties together a singleton TinTok token
//! and a set of TinRefs.  The TinRefs get safe "interior" mutability
//! by inheritance from the TinTok with the same invariant lifetime
//! parameter.  No compiler checks are bypassed.  This really is
//! inherited mutability.  The mutable XOR shared rule is still
//! enforced at compile time.  But you get all of the benefits of
//! interior mutability: it is safe to get a mutable reference to any
//! object which you have a TinRef to, even if the TinRef isn't itself
//! mutable.  You just need a mutable ref to the TinTok, which you can
//! (and should) pass around.  You can make cyclic graphs.  Bonus:
//! TinRefs are Sync + Send + Copy.
//!
//! The singleton-ness of the TinTok<'i> token is ensured by the
//! with_token function.  There is no other way users can construct
//! TinTok tokens.  The singleton-ness of the TinTok<'i> token in turn
//! ensures the mutable XOR shared property for & references created
//! from TinRefs<'i..>.  Note that there can be multiple TinToks in
//! use at the same time, in the same code.  They will necessarily
//! have distinct invariant lifetime parameters, even if references to
//! them have the same lifetime.  This can happen with nested
//! with_token calls.  TinTok<'i> for each 'i remains a singleton.
//!
//! The TinRefs have another lifetime parameter 'r (the reference
//! lifetime), in which they are covariant.  This lifetime was
//! borrowed from a &'r mut T at the time the TinRef<'i, 'r, T> was
//! created.  This lifetime prevents the TinRef from being used to
//! create dangling refrences to its target.  It also enforces the
//! no-aliasing rule by borrowing from the &'r mut T from which the
//! TinRef was created so that this &'r mut T cannot be used again.
//! Note the signatures of TinRef::get and TinRef::get_mut.
//!
//! You may notice that TinRefs share properties with vectors and
//! indices.  A (mutable|shared) reference to the TinTok<'i> token
//! acts like a (mutable|shared) reference to a vector head, and the
//! TinRefs<'i..> act like indices to that vector.  A reference
//! created from a TinRef<'i> borrows from a reference to the
//! TinTok<'> singleton, just as a reference to an element of a vector
//! borrows from the reference to the vector head.
//!
//! TinRef usage has no overhead above normal & reference usage.  They
//! have the null pointer optimization, so that Option<TinRef<..>>s
//! are the same size as TinRef<..>s, which are the same size as
//! normal & references.  TinRefs do not own their target object
//! (unlike Box) or share ownership (unlike Rc).  TinRefs are Copy +
//! Sync + Share.  TinRefs are designed for use with arena allocation,
//! where a group of objects are allocated separately, but are
//! deallocated together (or not at all).
//!
//! Unless using the functions get_many_maybe_mut, get_many_mut_maybe,
//! or get_many_mut, all of which do runtime checks for uniqueness,
//! there is only ever one mutable reference under the "control" of a
//! single TinTok<'i>, or multiple shared references.
//!
//! (Note on the syntactic sourness: maybe there is a way to use
//! macros to reduce the amount of get(tok) and get_mut(tok)
//! repititions?  Like maybe:
//!
//! a.get(tok).b.get(tok).c.get_mut(tok).d = e.get(tok).f;
//!
//! could be written like:
//!
//! tin!{tok| a-.b-.c+.d = e-.f};
//!
//! or something?)
//!
//! "Tin Refs, Rustie!"
//!
#[macro_use]
extern crate static_assertions;

use core::array;
use core::cmp::{PartialEq, PartialOrd};
use core::fmt::{Debug, Formatter, Pointer, Result};
use core::hash::Hash;
use core::marker::PhantomData;
use core::ptr::NonNull;

#[derive(Default, Copy, Clone, Hash, PartialOrd, PartialEq, Eq, Debug)]
struct Invariant<'i> {
    _i: PhantomData<*mut &'i ()>,
}

#[derive(Default, Copy, Clone, Hash, PartialOrd, PartialEq, Eq, Debug)]
struct Covariant<'c> {
    _c: PhantomData<&'c ()>,
}

#[derive(Default, Copy, Clone, Hash, PartialOrd, PartialEq, Eq, Debug)]
struct CoInvariant<'c, 'i> {
    _c: Covariant<'c>,
    _i: Invariant<'i>,
}

// Tin<'i>, unlike TinTok<'i> is Copy.  Its purpose is to be a
// convenient way to pass (by Copy) around the 'i invariant lifetime
// where needed, without using a reference to TinTok<'i>, which would
// interfere with TinRef usage.  Always copy Tin, never reference it!
#[derive(Default, Copy, Clone, Hash, PartialOrd, PartialEq, Eq, Debug)]
pub struct Tin<'i> {
    _i: Invariant<'i>,
}

pub struct TinTok<'i> {
    pub tin: Tin<'i>,
    _i: Invariant<'i>, // needs a private field
}

assert_not_impl_any!(TinTok: Copy, Clone); // stay single!

#[derive(Hash, PartialOrd, Eq, Debug)]
pub struct TinRef<'i, 'r, T: ?Sized + 'r> {
    p: NonNull<T>,
    _ci: CoInvariant<'r, 'i>,
}

// TinRefs are like vector indices: Copy, Clone, Send and Sync. The
// safety of & reference creation from them is preserved by requiring
// a matching TinTok<'i>, which is !Copy and !Clone, thus forever a
// singleton.
impl<'i, 'r, T: ?Sized + 'r> Copy for TinRef<'i, 'r, T> {}

impl<'i, 'r, T: ?Sized + 'r> Clone for TinRef<'i, 'r, T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

assert_not_impl_any!([()]: Copy, Clone);
assert_impl_all!(TinRef<'_, '_, [()]>: Copy, Clone, Sync, Send);

assert_eq_size!(TinRef<u32>, usize); // pointer sized
assert_eq_align!(TinRef<u8>, usize); // pointer aligned
assert_eq_size!(Option<TinRef<u32>>, usize); // null pointer optimization

// "Canning" a mutable reference into a TinRef, which consumes the
// mutable reference.  This version needs to be called using a messy
// generic parameter list like so: `can::<'i, '_, _>(...)`.  So, try
// the tin::can version instead.  Or use From/into.
#[inline(always)]
pub fn can<'i, 'r, T: ?Sized + 'r>(reference: &'r mut T) -> TinRef<'i, 'r, T> {
    TinRef {
        p: NonNull::from(reference),
        _ci: Default::default(),
    }
}

// let tok : &mut TinTok<'i> = ...;
// let tin = tok.tin;
// let reference : &'r mut T = ...;
// let tinref = tin.can(reference);
//
// use `tin.can(reference)` on a copy of tin because Tin is Copy, so
// doesn't borrow any reference from tok to can:
impl<'i> Tin<'i> {
    #[inline(always)]
    pub fn can<'r, T: ?Sized + 'r>(self, reference: &'r mut T) -> TinRef<'i, 'r, T> {
        // Note that "self" must not be a reference, because we don't
        // want to borrow.
        can(reference)
    }
}

// Or, just use reference.into(), if the target is suitably typed.
//
// let tinref : TinTok<'i, 'r, T> = reference.into();
impl<'i, 'r, T: ?Sized + 'r> From<&'r mut T> for TinRef<'i, 'r, T> {
    #[inline(always)]
    fn from(reference: &'r mut T) -> Self {
        can(reference)
    }
}

// The only way a user of this module can create a TinTok<'i> is by
// supplying a closure to this with_token function.  In nested calls
// to with_token, there will be multiple TinToks, but they will each
// have a different invariant lifetime parameter, with respect to
// which they are singletons.
#[inline]
pub fn with_token<F, T>(f: F) -> T
where
    F: for<'i> FnOnce(&mut TinTok<'i>) -> T,
{
    let mut tok = TinTok {
        tin: Default::default(),
        _i: Default::default(),
    };
    f(&mut tok)
}

impl<'i, 'r, T: ?Sized + 'r> TinRef<'i, 'r, T> {
    #[inline(always)]
    unsafe fn as_ref(self) -> &'r T {
        self.p.as_ref()
    }

    #[inline(always)]
    unsafe fn as_mut(self) -> &'r mut T {
        &mut *self.p.as_ptr()
    }

    // SAFETY: shared ref borrow inherited from the singleton
    // TinTok<'i>
    #[inline(always)]
    pub fn get(self, _tok: &'r TinTok<'i>) -> &'r T {
        unsafe { self.as_ref() }
    }

    // SAFETY: mutable ref borrow inherited from the singleton
    // TinTok<'i>
    #[inline(always)]
    pub fn get_mut(self, _tok: &'r mut TinTok<'i>) -> &'r mut T {
        unsafe { self.as_mut() }
    }
}

// SAFETY: TinTok is !Copy and has no state - TinRef<'i..>s stay safe
// as long as &mut TinTok<'i> is exclusive ref to singleton
unsafe impl<'i> Sync for TinTok<'i> {}
unsafe impl<'i> Send for TinTok<'i> {}

// SAFETY: the singleton-ness of TinTok<'i> prevents mutable aliasing
// TinRefs across threads
unsafe impl<'i, 'r, T: ?Sized + 'r> Sync for TinRef<'i, 'r, T> {}
unsafe impl<'i, 'r, T: ?Sized + 'r> Send for TinRef<'i, 'r, T> {}

impl<'i, 'r, T: ?Sized + 'r> Pointer for TinRef<'i, 'r, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Pointer::fmt(&self.p, f)
    }
}

impl<'i, 'r, T: ?Sized + 'r> PartialEq for TinRef<'i, 'r, T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.p == other.p
    }
}

impl<'i> TinTok<'i> {
    // The safety of these get_many function variants is based on the
    // same properties as the safety of TinRef::get_mut - note the
    // similarity of their signatures.  The additional property needed
    // for safety is uniqueness of the TinRefs in the rs argument,
    // which is runtime checked.

    // Note: It may be worth having a special case for N > something
    // that uses HashMaps to find duplicates instead of the current
    // O(N^2) methods.  But most useful cases will have small N.

    pub fn get_many_maybe_mut<'r, T: ?Sized + 'r, const N: usize>(
        &'r mut self,
        rs: &[TinRef<'i, 'r, T>; N],
    ) -> [Option<&'r mut T>; N] {
        // element is None iff a duplicate of a later element
        array::from_fn(|i| {
            let e = rs[i];
            if rs[(i + 1)..].contains(&e) {
                None
            } else {
                // SAFETY: mutable borrow of TinTok<'i> singleton
                // (self), with no duplicates due to check above
                Some(unsafe { e.as_mut() })
            }
        })
    }

    pub fn get_many_mut_maybe<'r, T: ?Sized + 'r, const N: usize>(
        &'r mut self,
        rs: &[TinRef<'i, 'r, T>; N],
    ) -> [Option<&'r mut T>; N] {
        // element is None iff a duplicate of an earlier element
        array::from_fn(|i| {
            let e = rs[i];
            if rs[..i].contains(&e) {
                None
            } else {
                // SAFETY: mutable borrow of TinTok<'i> singelton
                // (self), with no duplicates due to check above
                Some(unsafe { e.as_mut() })
            }
        })
    }

    pub fn get_many_mut<'r, T: ?Sized + 'r, const N: usize>(
        &'r mut self,
        rs: &[TinRef<'i, 'r, T>; N],
    ) -> Option<[&'r mut T; N]> {
        // Yes, it's 2-pass, because core::array::try_from_fn is
        // currently only nightly.  This would be identical to
        // get_many_maybe_mut with try_from_fn instead of from_fn.
        if rs[1..].iter().enumerate().any(|(i, e)| rs[..i].contains(e)) {
            None
        } else {
            // SAFETY: mutable borrow of TinTok<'i> singleton (self),
            // with no duplicates due to check above
            Some(array::from_fn(|i| unsafe { rs[i].as_mut() }))
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    // Nothing much yet - just see if expected usage compiles.
    //
    // Probably use something like https://docs.rs/trybuild to test
    // that attempts to violate Rust mutable XOR shared and lifetime
    // rules don't compile.
    //
    // Also need runtime tests for the get_many_mut variants.
    //
    #![allow(unused)]
    use super::*;

    type NodeR<'i, 'r> = TinRef<'i, 'r, Node<'i, 'r>>;

    struct Node<'i, 'r> {
        x: i32,
        c1: Option<NodeR<'i, 'r>>,
        c2: Option<NodeR<'i, 'r>>,
    }

    impl<'i, 'r> Node<'i, 'r> {
        fn new(x: i32) -> Self {
            Node {
                x,
                c1: None,
                c2: None,
            }
        }
    }

    // We can write impls on NodeR instead of Node.  The advantage is
    // that we delay getting the ref to self (which borrows from tok)
    // until needed.
    impl<'i, 'r> NodeR<'i, 'r> {
        fn set_x(self, tok: &mut TinTok<'i>, y: i32) {
            self.get_mut(tok).x = y;
        }

        fn cycle_c2(self, tok: &mut TinTok<'i>) {
            // Demonstrate interior mutability by creating a
            // self-cycle when the only mutable thing we have is
            // &mut tok
            self.get_mut(tok).c2 = Some(self);
        }

        fn set_x_from_c1(self, tok: &mut TinTok<'i>) {
            let y = self.get(tok).c1.unwrap().get(tok).x;
            self.get_mut(tok).x = y;
        }

        fn make_c1(self, tok: &mut TinTok<'i>) {
            let tin = tok.tin;
            // Use your favorite arena instead:
            let node: &mut _ = Box::leak(Box::new(Node::new(42)));
            self.get_mut(tok).c1 = Some(tin.can(node));
        }
    }

    fn thread_test(r1: &mut i32, r2: &mut i32) {
        use std::sync::RwLock;
        use std::thread;

        with_token(|tok| {
            // Copy the tin out early, so it can be used to can refs
            // without borrowing from tok.  It is Copy + Send + Sync,
            // so after making the first copy, everything in this
            // scope can use it, including threads.
            let tin = tok.tin;
            let tr1 = tin.can(r1);
            let tr2 = tin.can(r2);

            let tok_rw_lock = RwLock::new(tok);
            let tok_rw_lock_ref = &tok_rw_lock;
            thread::scope(|s| {
                // Obviously, one of these two threads will block due
                // to the .write()s.  If we had done .reads(), we
                // would then only get shared refs to tok, so still
                // could do get()s but not get_mut().
                s.spawn(move || {
                    let mut tok_lock = tok_rw_lock_ref.write().unwrap();
                    let tok: &mut _ = *tok_lock;
                    *tr1.get_mut(tok) = *tr2.get(tok);
                });
                s.spawn(move || {
                    let mut tok_lock = tok_rw_lock_ref.write().unwrap();
                    let tok: &mut _ = *tok_lock;
                    *tr2.get_mut(tok) = *tr1.get(tok);
                });
            });

            println!("r1 = {}, r2 = {}", r1, r2);
        })
    }
}
