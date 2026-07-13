//! Internal BLAS/LAPACK dispatch for the four conventional scalar families.

use crate::LapackScalar;
use num_complex::{Complex32, Complex64};

pub(crate) trait TriangularPackedBackend: crate::LapackScalar {
    const IS_COMPLEX: bool;
    unsafe fn tpmv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[Self], x: &mut [Self], incx: i32);
    unsafe fn tpsv(uplo: u8, trans: u8, diag: u8, n: i32, ap: &[Self], x: &mut [Self], incx: i32);
    unsafe fn tptrs(
        uplo: u8,
        trans: u8,
        diag: u8,
        n: i32,
        nrhs: i32,
        ap: &[Self],
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    );
    unsafe fn tptri(uplo: u8, diag: u8, n: i32, ap: &mut [Self], info: &mut i32);
    unsafe fn tpcon(
        norm: u8,
        uplo: u8,
        diag: u8,
        n: i32,
        ap: &[Self],
        rcond: &mut Self::Real,
        work: &mut [Self],
        realwork: &mut [Self::Real],
        iwork: &mut [i32],
        info: &mut i32,
    );
    unsafe fn tprfs(
        uplo: u8,
        trans: u8,
        diag: u8,
        n: i32,
        nrhs: i32,
        ap: &[Self],
        b: &[Self],
        ldb: i32,
        x: &mut [Self],
        ldx: i32,
        ferr: &mut [Self::Real],
        berr: &mut [Self::Real],
        work: &mut [Self],
        realwork: &mut [Self::Real],
        iwork: &mut [i32],
        info: &mut i32,
    );
    unsafe fn lantp(
        norm: u8,
        uplo: u8,
        diag: u8,
        n: i32,
        ap: &[Self],
        work: &mut [Self::Real],
    ) -> Self::Real;
}

pub(crate) trait PositiveDefinitePackedBackend: crate::LapackScalar {
    const IS_COMPLEX: bool;
    unsafe fn pmv(
        uplo: u8,
        n: i32,
        alpha: Self,
        ap: &[Self],
        x: &[Self],
        beta: Self,
        y: &mut [Self],
    );
    unsafe fn pptrf(uplo: u8, n: i32, ap: &mut [Self], info: &mut i32);
    unsafe fn pptrs(
        uplo: u8,
        n: i32,
        nrhs: i32,
        ap: &[Self],
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    );
    unsafe fn pptri(uplo: u8, n: i32, ap: &mut [Self], info: &mut i32);
    unsafe fn ppcon(
        uplo: u8,
        n: i32,
        ap: &[Self],
        anorm: Self::Real,
        rcond: &mut Self::Real,
        work: &mut [Self],
        realwork: &mut [Self::Real],
        iwork: &mut [i32],
        info: &mut i32,
    );
    unsafe fn pprfs(
        uplo: u8,
        n: i32,
        nrhs: i32,
        ap: &[Self],
        afp: &[Self],
        b: &[Self],
        ldb: i32,
        x: &mut [Self],
        ldx: i32,
        ferr: &mut [Self::Real],
        berr: &mut [Self::Real],
        work: &mut [Self],
        realwork: &mut [Self::Real],
        iwork: &mut [i32],
        info: &mut i32,
    );
}

pub(crate) trait SymmetricPackedBackend: crate::LapackScalar {
    const IS_COMPLEX: bool;
    unsafe fn sptrf(uplo: u8, n: i32, ap: &mut [Self], ipiv: &mut [i32], info: &mut i32);
    unsafe fn sptrs(
        uplo: u8,
        n: i32,
        nrhs: i32,
        ap: &[Self],
        ipiv: &[i32],
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    );
    unsafe fn sptri(
        uplo: u8,
        n: i32,
        ap: &mut [Self],
        ipiv: &[i32],
        work: &mut [Self],
        info: &mut i32,
    );
    unsafe fn spcon(
        uplo: u8,
        n: i32,
        ap: &[Self],
        ipiv: &[i32],
        anorm: Self::Real,
        rcond: &mut Self::Real,
        work: &mut [Self],
        iwork: &mut [i32],
        info: &mut i32,
    );
    unsafe fn sprfs(
        uplo: u8,
        n: i32,
        nrhs: i32,
        ap: &[Self],
        afp: &[Self],
        ipiv: &[i32],
        b: &[Self],
        ldb: i32,
        x: &mut [Self],
        ldx: i32,
        ferr: &mut [Self::Real],
        berr: &mut [Self::Real],
        work: &mut [Self],
        realwork: &mut [Self::Real],
        iwork: &mut [i32],
        info: &mut i32,
    );
}

pub(crate) trait RealSymmetricPackedBlas: SymmetricPackedBackend {
    unsafe fn spmv(
        uplo: u8,
        n: i32,
        alpha: Self,
        ap: &[Self],
        x: &[Self],
        beta: Self,
        y: &mut [Self],
    );
}

pub(crate) trait SymmetricPackedEigen: LapackScalar<Real = Self> {
    unsafe fn spev(
        jobz: u8,
        uplo: u8,
        n: i32,
        ap: &mut [Self],
        w: &mut [Self],
        z: &mut [Self],
        ldz: i32,
        work: &mut [Self],
        info: &mut i32,
    );
}

pub(crate) trait HermitianPackedEigen: LapackScalar {
    unsafe fn hpev(
        jobz: u8,
        uplo: u8,
        n: i32,
        ap: &mut [Self],
        w: &mut [Self::Real],
        z: &mut [Self],
        ldz: i32,
        work: &mut [Self],
        rwork: &mut [Self::Real],
        info: &mut i32,
    );
}

pub(crate) trait SymmetricPackedDivideConquer: SymmetricPackedEigen {
    unsafe fn spevd(
        jobz: u8,
        uplo: u8,
        n: i32,
        ap: &mut [Self],
        w: &mut [Self],
        z: &mut [Self],
        ldz: i32,
        work: &mut [Self],
        lwork: i32,
        iwork: &mut [i32],
        liwork: i32,
        info: &mut i32,
    );
    fn workspace_recommendation(value: Self) -> Option<usize>;
}

pub(crate) trait HermitianPackedDivideConquer: HermitianPackedEigen {
    unsafe fn hpevd(
        jobz: u8,
        uplo: u8,
        n: i32,
        ap: &mut [Self],
        w: &mut [Self::Real],
        z: &mut [Self],
        ldz: i32,
        work: &mut [Self],
        lwork: i32,
        rwork: &mut [Self::Real],
        lrwork: i32,
        iwork: &mut [i32],
        liwork: i32,
        info: &mut i32,
    );
    fn work_recommendation(value: Self) -> Option<usize>;
    fn real_work_recommendation(value: Self::Real) -> Option<usize>;
}

pub(crate) trait SymmetricPackedSelected: SymmetricPackedEigen {
    unsafe fn spevx(
        jobz: u8,
        range: u8,
        uplo: u8,
        n: i32,
        ap: &mut [Self],
        vl: Self,
        vu: Self,
        il: i32,
        iu: i32,
        abstol: Self,
        m: &mut i32,
        w: &mut [Self],
        z: &mut [Self],
        ldz: i32,
        work: &mut [Self],
        iwork: &mut [i32],
        ifail: &mut [i32],
        info: &mut i32,
    );
    fn finite(value: Self) -> bool;
    fn less(left: Self, right: Self) -> bool;
}
pub(crate) trait HermitianPackedSelected: HermitianPackedEigen {
    unsafe fn hpevx(
        jobz: u8,
        range: u8,
        uplo: u8,
        n: i32,
        ap: &mut [Self],
        vl: Self::Real,
        vu: Self::Real,
        il: i32,
        iu: i32,
        abstol: Self::Real,
        m: &mut i32,
        w: &mut [Self::Real],
        z: &mut [Self],
        ldz: i32,
        work: &mut [Self],
        rwork: &mut [Self::Real],
        iwork: &mut [i32],
        ifail: &mut [i32],
        info: &mut i32,
    );
    fn finite(value: Self::Real) -> bool;
    fn less(left: Self::Real, right: Self::Real) -> bool;
}
macro_rules! impl_spevx {
    ($t:ty,$f:path) => {
        impl SymmetricPackedSelected for $t {
            unsafe fn spevx(
                j: u8,
                r: u8,
                u: u8,
                n: i32,
                ap: &mut [Self],
                vl: Self,
                vu: Self,
                il: i32,
                iu: i32,
                at: Self,
                m: &mut i32,
                w: &mut [Self],
                z: &mut [Self],
                ldz: i32,
                work: &mut [Self],
                iw: &mut [i32],
                fail: &mut [i32],
                info: &mut i32,
            ) {
                unsafe {
                    $f(
                        j, r, u, n, ap, vl, vu, il, iu, at, m, w, z, ldz, work, iw, fail, info,
                    )
                }
            }
            fn finite(v: Self) -> bool {
                v.is_finite()
            }
            fn less(a: Self, b: Self) -> bool {
                a < b
            }
        }
    };
}
impl_spevx!(f32, lapack::sspevx);
impl_spevx!(f64, lapack::dspevx);
macro_rules! impl_hpevx {
    ($t:ty,$f:path) => {
        impl HermitianPackedSelected for $t {
            unsafe fn hpevx(
                j: u8,
                r: u8,
                u: u8,
                n: i32,
                ap: &mut [Self],
                vl: Self::Real,
                vu: Self::Real,
                il: i32,
                iu: i32,
                at: Self::Real,
                m: &mut i32,
                w: &mut [Self::Real],
                z: &mut [Self],
                ldz: i32,
                work: &mut [Self],
                rw: &mut [Self::Real],
                iw: &mut [i32],
                fail: &mut [i32],
                info: &mut i32,
            ) {
                unsafe {
                    $f(
                        j, r, u, n, ap, vl, vu, il, iu, at, m, w, z, ldz, work, rw, iw, fail, info,
                    )
                }
            }
            fn finite(v: Self::Real) -> bool {
                v.is_finite()
            }
            fn less(a: Self::Real, b: Self::Real) -> bool {
                a < b
            }
        }
    };
}
impl_hpevx!(Complex32, lapack::chpevx);
impl_hpevx!(Complex64, lapack::zhpevx);

fn float_workspace(value: f64) -> Option<usize> {
    if value.is_finite() && value >= 1.0 && value.fract() == 0.0 && value <= i32::MAX as f64 {
        Some(value as usize)
    } else {
        None
    }
}
macro_rules! impl_spevd {
    ($t:ty,$f:path) => {
        impl SymmetricPackedDivideConquer for $t {
            unsafe fn spevd(
                j: u8,
                u: u8,
                n: i32,
                ap: &mut [Self],
                w: &mut [Self],
                z: &mut [Self],
                ldz: i32,
                work: &mut [Self],
                lw: i32,
                iw: &mut [i32],
                liw: i32,
                info: &mut i32,
            ) {
                unsafe { $f(j, u, n, ap, w, z, ldz, work, lw, iw, liw, info) }
            }
            fn workspace_recommendation(v: Self) -> Option<usize> {
                float_workspace(v as f64)
            }
        }
    };
}
impl_spevd!(f32, lapack::sspevd);
impl_spevd!(f64, lapack::dspevd);
macro_rules! impl_hpevd {
    ($t:ty,$f:path) => {
        impl HermitianPackedDivideConquer for $t {
            unsafe fn hpevd(
                j: u8,
                u: u8,
                n: i32,
                ap: &mut [Self],
                w: &mut [Self::Real],
                z: &mut [Self],
                ldz: i32,
                work: &mut [Self],
                lw: i32,
                rw: &mut [Self::Real],
                lrw: i32,
                iw: &mut [i32],
                liw: i32,
                info: &mut i32,
            ) {
                unsafe { $f(j, u, n, ap, w, z, ldz, work, lw, rw, lrw, iw, liw, info) }
            }
            fn work_recommendation(v: Self) -> Option<usize> {
                float_workspace(v.re as f64)
            }
            fn real_work_recommendation(v: Self::Real) -> Option<usize> {
                float_workspace(v as f64)
            }
        }
    };
}
impl_hpevd!(Complex32, lapack::chpevd);
impl_hpevd!(Complex64, lapack::zhpevd);

macro_rules! impl_spev {
    ($t:ty, $f:path) => {
        impl SymmetricPackedEigen for $t {
            unsafe fn spev(
                j: u8,
                u: u8,
                n: i32,
                ap: &mut [Self],
                w: &mut [Self],
                z: &mut [Self],
                ldz: i32,
                work: &mut [Self],
                info: &mut i32,
            ) {
                unsafe { $f(j, u, n, ap, w, z, ldz, work, info) }
            }
        }
    };
}
impl_spev!(f32, lapack::sspev);
impl_spev!(f64, lapack::dspev);

macro_rules! impl_hpev {
    ($t:ty, $f:path) => {
        impl HermitianPackedEigen for $t {
            unsafe fn hpev(
                j: u8,
                u: u8,
                n: i32,
                ap: &mut [Self],
                w: &mut [Self::Real],
                z: &mut [Self],
                ldz: i32,
                work: &mut [Self],
                rwork: &mut [Self::Real],
                info: &mut i32,
            ) {
                unsafe { $f(j, u, n, ap, w, z, ldz, work, rwork, info) }
            }
        }
    };
}
impl_hpev!(Complex32, lapack::chpev);
impl_hpev!(Complex64, lapack::zhpev);

pub(crate) trait HermitianPackedBackend: crate::LapackScalar {
    unsafe fn hpmv(
        uplo: u8,
        n: i32,
        alpha: Self,
        ap: &[Self],
        x: &[Self],
        beta: Self,
        y: &mut [Self],
    );
    unsafe fn hptrf(uplo: u8, n: i32, ap: &mut [Self], ipiv: &mut [i32], info: &mut i32);
    unsafe fn hptrs(
        uplo: u8,
        n: i32,
        nrhs: i32,
        ap: &[Self],
        ipiv: &[i32],
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    );
    unsafe fn hptri(
        uplo: u8,
        n: i32,
        ap: &mut [Self],
        ipiv: &[i32],
        work: &mut [Self],
        info: &mut i32,
    );
    unsafe fn hpcon(
        uplo: u8,
        n: i32,
        ap: &[Self],
        ipiv: &[i32],
        anorm: Self::Real,
        rcond: &mut Self::Real,
        work: &mut [Self],
        info: &mut i32,
    );
    unsafe fn hprfs(
        uplo: u8,
        n: i32,
        nrhs: i32,
        ap: &[Self],
        afp: &[Self],
        ipiv: &[i32],
        b: &[Self],
        ldb: i32,
        x: &mut [Self],
        ldx: i32,
        ferr: &mut [Self::Real],
        berr: &mut [Self::Real],
        work: &mut [Self],
        realwork: &mut [Self::Real],
        info: &mut i32,
    );
}

macro_rules! impl_triangular_real {
    ($t:ty, $mv:path, $sv:path, $trs:path, $tri:path, $con:path, $rfs:path, $lan:path) => {
        impl TriangularPackedBackend for $t {
            const IS_COMPLEX: bool = false;
            unsafe fn tpmv(
                uplo: u8,
                trans: u8,
                diag: u8,
                n: i32,
                ap: &[Self],
                x: &mut [Self],
                incx: i32,
            ) {
                unsafe { $mv(uplo, trans, diag, n, ap, x, incx) }
            }
            unsafe fn tpsv(
                uplo: u8,
                trans: u8,
                diag: u8,
                n: i32,
                ap: &[Self],
                x: &mut [Self],
                incx: i32,
            ) {
                unsafe { $sv(uplo, trans, diag, n, ap, x, incx) }
            }
            unsafe fn tptrs(
                uplo: u8,
                trans: u8,
                diag: u8,
                n: i32,
                nrhs: i32,
                ap: &[Self],
                b: &mut [Self],
                ldb: i32,
                info: &mut i32,
            ) {
                unsafe { $trs(uplo, trans, diag, n, nrhs, ap, b, ldb, info) }
            }
            unsafe fn tptri(uplo: u8, diag: u8, n: i32, ap: &mut [Self], info: &mut i32) {
                unsafe { $tri(uplo, diag, n, ap, info) }
            }
            unsafe fn tpcon(
                no: u8,
                u: u8,
                d: u8,
                n: i32,
                ap: &[Self],
                r: &mut Self::Real,
                w: &mut [Self],
                _rw: &mut [Self::Real],
                iw: &mut [i32],
                info: &mut i32,
            ) {
                unsafe { $con(no, u, d, n, ap, r, w, iw, info) }
            }
            unsafe fn tprfs(
                u: u8,
                t: u8,
                d: u8,
                n: i32,
                nr: i32,
                ap: &[Self],
                b: &[Self],
                ldb: i32,
                x: &mut [Self],
                ldx: i32,
                f: &mut [Self::Real],
                be: &mut [Self::Real],
                w: &mut [Self],
                _rw: &mut [Self::Real],
                iw: &mut [i32],
                info: &mut i32,
            ) {
                unsafe { $rfs(u, t, d, n, nr, ap, b, ldb, x, ldx, f, be, w, iw, info) }
            }
            unsafe fn lantp(
                no: u8,
                u: u8,
                d: u8,
                n: i32,
                ap: &[Self],
                w: &mut [Self::Real],
            ) -> Self::Real {
                unsafe { $lan(no, u, d, n, ap, w) }
            }
        }
    };
}
macro_rules! impl_triangular_complex {
    ($t:ty, $mv:path, $sv:path, $trs:path, $tri:path, $con:path, $rfs:path, $lan:path) => {
        impl TriangularPackedBackend for $t {
            const IS_COMPLEX: bool = true;
            unsafe fn tpmv(u: u8, t: u8, d: u8, n: i32, ap: &[Self], x: &mut [Self], incx: i32) {
                unsafe { $mv(u, t, d, n, ap, x, incx) }
            }
            unsafe fn tpsv(u: u8, t: u8, d: u8, n: i32, ap: &[Self], x: &mut [Self], incx: i32) {
                unsafe { $sv(u, t, d, n, ap, x, incx) }
            }
            unsafe fn tptrs(
                u: u8,
                t: u8,
                d: u8,
                n: i32,
                nr: i32,
                ap: &[Self],
                b: &mut [Self],
                ldb: i32,
                info: &mut i32,
            ) {
                unsafe { $trs(u, t, d, n, nr, ap, b, ldb, info) }
            }
            unsafe fn tptri(u: u8, d: u8, n: i32, ap: &mut [Self], info: &mut i32) {
                unsafe { $tri(u, d, n, ap, info) }
            }
            unsafe fn tpcon(
                no: u8,
                u: u8,
                d: u8,
                n: i32,
                ap: &[Self],
                r: &mut Self::Real,
                w: &mut [Self],
                rw: &mut [Self::Real],
                _iw: &mut [i32],
                info: &mut i32,
            ) {
                unsafe { $con(no, u, d, n, ap, r, w, rw, info) }
            }
            unsafe fn tprfs(
                u: u8,
                t: u8,
                d: u8,
                n: i32,
                nr: i32,
                ap: &[Self],
                b: &[Self],
                ldb: i32,
                x: &mut [Self],
                ldx: i32,
                f: &mut [Self::Real],
                be: &mut [Self::Real],
                w: &mut [Self],
                rw: &mut [Self::Real],
                _iw: &mut [i32],
                info: &mut i32,
            ) {
                unsafe { $rfs(u, t, d, n, nr, ap, b, ldb, x, ldx, f, be, w, rw, info) }
            }
            unsafe fn lantp(
                no: u8,
                u: u8,
                d: u8,
                n: i32,
                ap: &[Self],
                w: &mut [Self::Real],
            ) -> Self::Real {
                unsafe { $lan(no, u, d, n, ap, w) }
            }
        }
    };
}
impl_triangular_real!(
    f32,
    blas::stpmv,
    blas::stpsv,
    lapack::stptrs,
    lapack::stptri,
    lapack::stpcon,
    lapack::stprfs,
    lapack::slantp
);
impl_triangular_real!(
    f64,
    blas::dtpmv,
    blas::dtpsv,
    lapack::dtptrs,
    lapack::dtptri,
    lapack::dtpcon,
    lapack::dtprfs,
    lapack::dlantp
);
impl_triangular_complex!(
    Complex32,
    blas::ctpmv,
    blas::ctpsv,
    lapack::ctptrs,
    lapack::ctptri,
    lapack::ctpcon,
    lapack::ctprfs,
    lapack::clantp
);
impl_triangular_complex!(
    Complex64,
    blas::ztpmv,
    blas::ztpsv,
    lapack::ztptrs,
    lapack::ztptri,
    lapack::ztpcon,
    lapack::ztprfs,
    lapack::zlantp
);

macro_rules! impl_pd_real {
    ($t:ty, $mv:path, $trf:path, $trs:path, $tri:path, $con:path, $rfs:path) => {
        impl PositiveDefinitePackedBackend for $t {
            const IS_COMPLEX: bool = false;
            unsafe fn pmv(
                uplo: u8,
                n: i32,
                alpha: Self,
                ap: &[Self],
                x: &[Self],
                beta: Self,
                y: &mut [Self],
            ) {
                unsafe { $mv(uplo, n, alpha, ap, x, 1, beta, y, 1) }
            }
            unsafe fn pptrf(uplo: u8, n: i32, ap: &mut [Self], info: &mut i32) {
                unsafe { $trf(uplo, n, ap, info) }
            }
            unsafe fn pptrs(
                uplo: u8,
                n: i32,
                nrhs: i32,
                ap: &[Self],
                b: &mut [Self],
                ldb: i32,
                info: &mut i32,
            ) {
                unsafe { $trs(uplo, n, nrhs, ap, b, ldb, info) }
            }
            unsafe fn pptri(uplo: u8, n: i32, ap: &mut [Self], info: &mut i32) {
                unsafe { $tri(uplo, n, ap, info) }
            }
            unsafe fn ppcon(
                u: u8,
                n: i32,
                ap: &[Self],
                an: Self::Real,
                r: &mut Self::Real,
                w: &mut [Self],
                _rw: &mut [Self::Real],
                iw: &mut [i32],
                info: &mut i32,
            ) {
                unsafe { $con(u, n, ap, an, r, w, iw, info) }
            }
            unsafe fn pprfs(
                u: u8,
                n: i32,
                nr: i32,
                ap: &[Self],
                afp: &[Self],
                b: &[Self],
                ldb: i32,
                x: &mut [Self],
                ldx: i32,
                f: &mut [Self::Real],
                be: &mut [Self::Real],
                w: &mut [Self],
                _rw: &mut [Self::Real],
                iw: &mut [i32],
                info: &mut i32,
            ) {
                unsafe { $rfs(u, n, nr, ap, afp, b, ldb, x, ldx, f, be, w, iw, info) }
            }
        }
    };
}
impl_pd_real!(
    f32,
    blas::sspmv,
    lapack::spptrf,
    lapack::spptrs,
    lapack::spptri,
    lapack::sppcon,
    lapack::spprfs
);
impl_pd_real!(
    f64,
    blas::dspmv,
    lapack::dpptrf,
    lapack::dpptrs,
    lapack::dpptri,
    lapack::dppcon,
    lapack::dpprfs
);

macro_rules! impl_pd_complex {
    ($t:ty, $mv:path, $trf:path, $trs:path, $tri:path, $con:path, $rfs:path) => {
        impl PositiveDefinitePackedBackend for $t {
            const IS_COMPLEX: bool = true;
            unsafe fn pmv(
                uplo: u8,
                n: i32,
                alpha: Self,
                ap: &[Self],
                x: &[Self],
                beta: Self,
                y: &mut [Self],
            ) {
                unsafe { $mv(uplo, n, alpha, ap, x, 1, beta, y, 1) }
            }
            unsafe fn pptrf(uplo: u8, n: i32, ap: &mut [Self], info: &mut i32) {
                unsafe { $trf(uplo, n, ap, info) }
            }
            unsafe fn pptrs(
                uplo: u8,
                n: i32,
                nrhs: i32,
                ap: &[Self],
                b: &mut [Self],
                ldb: i32,
                info: &mut i32,
            ) {
                unsafe { $trs(uplo, n, nrhs, ap, b, ldb, info) }
            }
            unsafe fn pptri(uplo: u8, n: i32, ap: &mut [Self], info: &mut i32) {
                unsafe { $tri(uplo, n, ap, info) }
            }
            unsafe fn ppcon(
                u: u8,
                n: i32,
                ap: &[Self],
                an: Self::Real,
                r: &mut Self::Real,
                w: &mut [Self],
                rw: &mut [Self::Real],
                _iw: &mut [i32],
                info: &mut i32,
            ) {
                unsafe { $con(u, n, ap, an, r, w, rw, info) }
            }
            unsafe fn pprfs(
                u: u8,
                n: i32,
                nr: i32,
                ap: &[Self],
                afp: &[Self],
                b: &[Self],
                ldb: i32,
                x: &mut [Self],
                ldx: i32,
                f: &mut [Self::Real],
                be: &mut [Self::Real],
                w: &mut [Self],
                rw: &mut [Self::Real],
                _iw: &mut [i32],
                info: &mut i32,
            ) {
                unsafe { $rfs(u, n, nr, ap, afp, b, ldb, x, ldx, f, be, w, rw, info) }
            }
        }
    };
}
impl_pd_complex!(
    Complex32,
    blas::chpmv,
    lapack::cpptrf,
    lapack::cpptrs,
    lapack::cpptri,
    lapack::cppcon,
    lapack::cpprfs
);
impl_pd_complex!(
    Complex64,
    blas::zhpmv,
    lapack::zpptrf,
    lapack::zpptrs,
    lapack::zpptri,
    lapack::zppcon,
    lapack::zpprfs
);

macro_rules! impl_sym_real {
    ($t:ty, $trf:path, $trs:path, $tri:path, $con:path, $rfs:path) => {
        impl SymmetricPackedBackend for $t {
            const IS_COMPLEX: bool = false;
            unsafe fn sptrf(uplo: u8, n: i32, ap: &mut [Self], ipiv: &mut [i32], info: &mut i32) {
                unsafe { $trf(uplo, n, ap, ipiv, info) }
            }
            unsafe fn sptrs(
                uplo: u8,
                n: i32,
                nrhs: i32,
                ap: &[Self],
                ipiv: &[i32],
                b: &mut [Self],
                ldb: i32,
                info: &mut i32,
            ) {
                unsafe { $trs(uplo, n, nrhs, ap, ipiv, b, ldb, info) }
            }
            unsafe fn sptri(
                uplo: u8,
                n: i32,
                ap: &mut [Self],
                ipiv: &[i32],
                work: &mut [Self],
                info: &mut i32,
            ) {
                unsafe { $tri(uplo, n, ap, ipiv, work, info) }
            }
            unsafe fn spcon(
                u: u8,
                n: i32,
                ap: &[Self],
                ipiv: &[i32],
                an: Self::Real,
                r: &mut Self::Real,
                w: &mut [Self],
                iw: &mut [i32],
                info: &mut i32,
            ) {
                unsafe { $con(u, n, ap, ipiv, an, r, w, iw, info) }
            }
            unsafe fn sprfs(
                u: u8,
                n: i32,
                nr: i32,
                ap: &[Self],
                afp: &[Self],
                ip: &[i32],
                b: &[Self],
                ldb: i32,
                x: &mut [Self],
                ldx: i32,
                f: &mut [Self::Real],
                be: &mut [Self::Real],
                w: &mut [Self],
                _rw: &mut [Self::Real],
                iw: &mut [i32],
                info: &mut i32,
            ) {
                unsafe { $rfs(u, n, nr, ap, afp, ip, b, ldb, x, ldx, f, be, w, iw, info) }
            }
        }
    };
}
impl_sym_real!(
    f32,
    lapack::ssptrf,
    lapack::ssptrs,
    lapack::ssptri,
    lapack::sspcon,
    lapack::ssprfs
);
impl_sym_real!(
    f64,
    lapack::dsptrf,
    lapack::dsptrs,
    lapack::dsptri,
    lapack::dspcon,
    lapack::dsprfs
);

macro_rules! impl_sym_complex {
    ($t:ty, $trf:path, $trs:path, $tri:path, $con:path, $rfs:path) => {
        impl SymmetricPackedBackend for $t {
            const IS_COMPLEX: bool = true;
            unsafe fn sptrf(uplo: u8, n: i32, ap: &mut [Self], ipiv: &mut [i32], info: &mut i32) {
                unsafe { $trf(uplo, n, ap, ipiv, info) }
            }
            unsafe fn sptrs(
                uplo: u8,
                n: i32,
                nrhs: i32,
                ap: &[Self],
                ipiv: &[i32],
                b: &mut [Self],
                ldb: i32,
                info: &mut i32,
            ) {
                unsafe { $trs(uplo, n, nrhs, ap, ipiv, b, ldb, info) }
            }
            unsafe fn sptri(
                uplo: u8,
                n: i32,
                ap: &mut [Self],
                ipiv: &[i32],
                work: &mut [Self],
                info: &mut i32,
            ) {
                unsafe { $tri(uplo, n, ap, ipiv, work, info) }
            }
            unsafe fn spcon(
                u: u8,
                n: i32,
                ap: &[Self],
                ipiv: &[i32],
                an: Self::Real,
                r: &mut Self::Real,
                w: &mut [Self],
                _iw: &mut [i32],
                info: &mut i32,
            ) {
                unsafe { $con(u, n, ap, ipiv, an, r, w, info) }
            }
            unsafe fn sprfs(
                u: u8,
                n: i32,
                nr: i32,
                ap: &[Self],
                afp: &[Self],
                ip: &[i32],
                b: &[Self],
                ldb: i32,
                x: &mut [Self],
                ldx: i32,
                f: &mut [Self::Real],
                be: &mut [Self::Real],
                w: &mut [Self],
                rw: &mut [Self::Real],
                _iw: &mut [i32],
                info: &mut i32,
            ) {
                unsafe { $rfs(u, n, nr, ap, afp, ip, b, ldb, x, ldx, f, be, w, rw, info) }
            }
        }
    };
}
impl_sym_complex!(
    Complex32,
    lapack::csptrf,
    lapack::csptrs,
    lapack::csptri,
    lapack::cspcon,
    lapack::csprfs
);
impl_sym_complex!(
    Complex64,
    lapack::zsptrf,
    lapack::zsptrs,
    lapack::zsptri,
    lapack::zspcon,
    lapack::zsprfs
);

impl RealSymmetricPackedBlas for f32 {
    unsafe fn spmv(u: u8, n: i32, a: Self, ap: &[Self], x: &[Self], b: Self, y: &mut [Self]) {
        unsafe { blas::sspmv(u, n, a, ap, x, 1, b, y, 1) }
    }
}
impl RealSymmetricPackedBlas for f64 {
    unsafe fn spmv(u: u8, n: i32, a: Self, ap: &[Self], x: &[Self], b: Self, y: &mut [Self]) {
        unsafe { blas::dspmv(u, n, a, ap, x, 1, b, y, 1) }
    }
}

macro_rules! impl_herm {
    ($t:ty, $mv:path, $trf:path, $trs:path, $tri:path, $con:path, $rfs:path) => {
        impl HermitianPackedBackend for $t {
            unsafe fn hpmv(
                u: u8,
                n: i32,
                a: Self,
                ap: &[Self],
                x: &[Self],
                b: Self,
                y: &mut [Self],
            ) {
                unsafe { $mv(u, n, a, ap, x, 1, b, y, 1) }
            }
            unsafe fn hptrf(u: u8, n: i32, ap: &mut [Self], ipiv: &mut [i32], info: &mut i32) {
                unsafe { $trf(u, n, ap, ipiv, info) }
            }
            unsafe fn hptrs(
                u: u8,
                n: i32,
                nrhs: i32,
                ap: &[Self],
                ipiv: &[i32],
                b: &mut [Self],
                ldb: i32,
                info: &mut i32,
            ) {
                unsafe { $trs(u, n, nrhs, ap, ipiv, b, ldb, info) }
            }
            unsafe fn hptri(
                u: u8,
                n: i32,
                ap: &mut [Self],
                ipiv: &[i32],
                work: &mut [Self],
                info: &mut i32,
            ) {
                unsafe { $tri(u, n, ap, ipiv, work, info) }
            }
            unsafe fn hpcon(
                u: u8,
                n: i32,
                ap: &[Self],
                ipiv: &[i32],
                an: Self::Real,
                r: &mut Self::Real,
                w: &mut [Self],
                info: &mut i32,
            ) {
                unsafe { $con(u, n, ap, ipiv, an, r, w, info) }
            }
            unsafe fn hprfs(
                u: u8,
                n: i32,
                nr: i32,
                ap: &[Self],
                afp: &[Self],
                ip: &[i32],
                b: &[Self],
                ldb: i32,
                x: &mut [Self],
                ldx: i32,
                f: &mut [Self::Real],
                be: &mut [Self::Real],
                w: &mut [Self],
                rw: &mut [Self::Real],
                info: &mut i32,
            ) {
                unsafe { $rfs(u, n, nr, ap, afp, ip, b, ldb, x, ldx, f, be, w, rw, info) }
            }
        }
    };
}
impl_herm!(
    Complex32,
    blas::chpmv,
    lapack::chptrf,
    lapack::chptrs,
    lapack::chptri,
    lapack::chpcon,
    lapack::chprfs
);
impl_herm!(
    Complex64,
    blas::zhpmv,
    lapack::zhptrf,
    lapack::zhptrs,
    lapack::zhptri,
    lapack::zhpcon,
    lapack::zhprfs
);
