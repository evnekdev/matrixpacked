//! Internal BLAS/LAPACK dispatch for the four conventional scalar families.

use num_complex::{Complex32, Complex64};

pub(crate) trait TriangularPackedBackend: crate::LapackScalar {
    unsafe fn tpmv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[Self], x: &mut [Self]);
    unsafe fn tptrs(uplo: u8, trans: u8, diag: u8, n: i32, nrhs: i32, ap: &[Self], b: &mut [Self], ldb: i32, info: &mut i32);
    unsafe fn tptri(uplo: u8, diag: u8, n: i32, ap: &mut [Self], info: &mut i32);
}

pub(crate) trait PositiveDefinitePackedBackend: crate::LapackScalar {
    unsafe fn pmv(uplo: u8, n: i32, alpha: Self, ap: &[Self], x: &[Self], beta: Self, y: &mut [Self]);
    unsafe fn pptrf(uplo: u8, n: i32, ap: &mut [Self], info: &mut i32);
    unsafe fn pptrs(uplo: u8, n: i32, nrhs: i32, ap: &[Self], b: &mut [Self], ldb: i32, info: &mut i32);
    unsafe fn pptri(uplo: u8, n: i32, ap: &mut [Self], info: &mut i32);
}

pub(crate) trait SymmetricPackedBackend: crate::LapackScalar {
    unsafe fn sptrf(uplo: u8, n: i32, ap: &mut [Self], ipiv: &mut [i32], info: &mut i32);
    unsafe fn sptrs(uplo: u8, n: i32, nrhs: i32, ap: &[Self], ipiv: &[i32], b: &mut [Self], ldb: i32, info: &mut i32);
    unsafe fn sptri(uplo: u8, n: i32, ap: &mut [Self], ipiv: &[i32], work: &mut [Self], info: &mut i32);
}

pub(crate) trait RealSymmetricPackedBlas: SymmetricPackedBackend {
    unsafe fn spmv(uplo: u8, n: i32, alpha: Self, ap: &[Self], x: &[Self], beta: Self, y: &mut [Self]);
}

pub(crate) trait HermitianPackedBackend: crate::LapackScalar {
    unsafe fn hpmv(uplo: u8, n: i32, alpha: Self, ap: &[Self], x: &[Self], beta: Self, y: &mut [Self]);
    unsafe fn hptrf(uplo: u8, n: i32, ap: &mut [Self], ipiv: &mut [i32], info: &mut i32);
    unsafe fn hptrs(uplo: u8, n: i32, nrhs: i32, ap: &[Self], ipiv: &[i32], b: &mut [Self], ldb: i32, info: &mut i32);
    unsafe fn hptri(uplo: u8, n: i32, ap: &mut [Self], ipiv: &[i32], work: &mut [Self], info: &mut i32);
}

macro_rules! impl_triangular {
    ($t:ty, $mv:path, $trs:path, $tri:path) => {
        impl TriangularPackedBackend for $t {
            unsafe fn tpmv(uplo:u8, trans:u8, diag:u8, n:i32, ap:&[Self], x:&mut[Self]) { unsafe { $mv(uplo, trans, diag, n, ap, x, 1) } }
            unsafe fn tptrs(uplo:u8, trans:u8, diag:u8, n:i32, nrhs:i32, ap:&[Self], b:&mut[Self], ldb:i32, info:&mut i32) { unsafe { $trs(uplo, trans, diag, n, nrhs, ap, b, ldb, info) } }
            unsafe fn tptri(uplo:u8, diag:u8, n:i32, ap:&mut[Self], info:&mut i32) { unsafe { $tri(uplo,diag,n,ap,info) } }
        }
    };
}
impl_triangular!(f32, blas::stpmv, lapack::stptrs, lapack::stptri);
impl_triangular!(f64, blas::dtpmv, lapack::dtptrs, lapack::dtptri);
impl_triangular!(Complex32, blas::ctpmv, lapack::ctptrs, lapack::ctptri);
impl_triangular!(Complex64, blas::ztpmv, lapack::ztptrs, lapack::ztptri);

macro_rules! impl_pd_real {
    ($t:ty, $mv:path, $trf:path, $trs:path, $tri:path) => {
        impl PositiveDefinitePackedBackend for $t {
            unsafe fn pmv(uplo:u8,n:i32,alpha:Self,ap:&[Self],x:&[Self],beta:Self,y:&mut[Self]) { unsafe { $mv(uplo,n,alpha,ap,x,1,beta,y,1) } }
            unsafe fn pptrf(uplo:u8,n:i32,ap:&mut[Self],info:&mut i32) { unsafe { $trf(uplo,n,ap,info) } }
            unsafe fn pptrs(uplo:u8,n:i32,nrhs:i32,ap:&[Self],b:&mut[Self],ldb:i32,info:&mut i32) { unsafe { $trs(uplo,n,nrhs,ap,b,ldb,info) } }
            unsafe fn pptri(uplo:u8,n:i32,ap:&mut[Self],info:&mut i32) { unsafe { $tri(uplo,n,ap,info) } }
        }
    };
}
impl_pd_real!(f32, blas::sspmv, lapack::spptrf, lapack::spptrs, lapack::spptri);
impl_pd_real!(f64, blas::dspmv, lapack::dpptrf, lapack::dpptrs, lapack::dpptri);
impl_pd_real!(Complex32, blas::chpmv, lapack::cpptrf, lapack::cpptrs, lapack::cpptri);
impl_pd_real!(Complex64, blas::zhpmv, lapack::zpptrf, lapack::zpptrs, lapack::zpptri);

macro_rules! impl_sym {
    ($t:ty, $trf:path, $trs:path, $tri:path) => {
        impl SymmetricPackedBackend for $t {
            unsafe fn sptrf(uplo:u8,n:i32,ap:&mut[Self],ipiv:&mut[i32],info:&mut i32) { unsafe { $trf(uplo,n,ap,ipiv,info) } }
            unsafe fn sptrs(uplo:u8,n:i32,nrhs:i32,ap:&[Self],ipiv:&[i32],b:&mut[Self],ldb:i32,info:&mut i32) { unsafe { $trs(uplo,n,nrhs,ap,ipiv,b,ldb,info) } }
            unsafe fn sptri(uplo:u8,n:i32,ap:&mut[Self],ipiv:&[i32],work:&mut[Self],info:&mut i32) { unsafe { $tri(uplo,n,ap,ipiv,work,info) } }
        }
    };
}
impl_sym!(f32, lapack::ssptrf, lapack::ssptrs, lapack::ssptri);
impl_sym!(f64, lapack::dsptrf, lapack::dsptrs, lapack::dsptri);
impl_sym!(Complex32, lapack::csptrf, lapack::csptrs, lapack::csptri);
impl_sym!(Complex64, lapack::zsptrf, lapack::zsptrs, lapack::zsptri);

impl RealSymmetricPackedBlas for f32 { unsafe fn spmv(u:u8,n:i32,a:Self,ap:&[Self],x:&[Self],b:Self,y:&mut[Self]) { unsafe { blas::sspmv(u,n,a,ap,x,1,b,y,1) } } }
impl RealSymmetricPackedBlas for f64 { unsafe fn spmv(u:u8,n:i32,a:Self,ap:&[Self],x:&[Self],b:Self,y:&mut[Self]) { unsafe { blas::dspmv(u,n,a,ap,x,1,b,y,1) } } }

macro_rules! impl_herm {
    ($t:ty, $mv:path, $trf:path, $trs:path, $tri:path) => {
        impl HermitianPackedBackend for $t {
            unsafe fn hpmv(u:u8,n:i32,a:Self,ap:&[Self],x:&[Self],b:Self,y:&mut[Self]) { unsafe { $mv(u,n,a,ap,x,1,b,y,1) } }
            unsafe fn hptrf(u:u8,n:i32,ap:&mut[Self],ipiv:&mut[i32],info:&mut i32) { unsafe { $trf(u,n,ap,ipiv,info) } }
            unsafe fn hptrs(u:u8,n:i32,nrhs:i32,ap:&[Self],ipiv:&[i32],b:&mut[Self],ldb:i32,info:&mut i32) { unsafe { $trs(u,n,nrhs,ap,ipiv,b,ldb,info) } }
            unsafe fn hptri(u:u8,n:i32,ap:&mut[Self],ipiv:&[i32],work:&mut[Self],info:&mut i32) { unsafe { $tri(u,n,ap,ipiv,work,info) } }
        }
    };
}
impl_herm!(Complex32, blas::chpmv, lapack::chptrf, lapack::chptrs, lapack::chptri);
impl_herm!(Complex64, blas::zhpmv, lapack::zhptrf, lapack::zhptrs, lapack::zhptri);
