use std::fmt;

/// Formats a logical square matrix using nalgebra-style aligned Unicode borders.
pub(crate) fn display_square<T, F>(
    formatter: &mut fmt::Formatter<'_>,
    n: usize,
    mut get: F,
) -> fmt::Result
where
    T: fmt::Display,
    F: FnMut(usize, usize) -> T,
{
    if n == 0 {
        return formatter.write_str("[]");
    }

    let precision = formatter.precision();
    let mut values = Vec::with_capacity(n * n);
    let mut widths = vec![0usize; n];

    for row in 0..n {
        for col in 0..n {
            let value = get(row, col);
            let rendered = match precision {
                Some(precision) => format!("{value:.precision$}"),
                None => format!("{value}"),
            };
            widths[col] = widths[col].max(rendered.chars().count());
            values.push(rendered);
        }
    }

    let interior_width = widths.iter().sum::<usize>() + 2 * n + n.saturating_sub(1);
    writeln!(formatter, "  ┌{}┐", " ".repeat(interior_width))?;

    for row in 0..n {
        formatter.write_str("  │")?;
        for col in 0..n {
            let value = &values[row * n + col];
            write!(formatter, " {:>width$} ", value, width = widths[col])?;
            if col + 1 != n {
                formatter.write_str(" ")?;
            }
        }
        writeln!(formatter, "│")?;
    }

    write!(formatter, "  └{}┘", " ".repeat(interior_width))
}

/// Formats a logical square matrix like nalgebra's `Debug` output: a list of
/// columns, reflecting nalgebra's column-major storage convention.
pub(crate) fn debug_square<T, F>(
    formatter: &mut fmt::Formatter<'_>,
    n: usize,
    get: F,
) -> fmt::Result
where
    T: fmt::Debug,
    F: Fn(usize, usize) -> T,
{
    struct Column<'a, T, F> {
        col: usize,
        n: usize,
        get: &'a F,
        marker: std::marker::PhantomData<T>,
    }

    impl<T, F> fmt::Debug for Column<'_, T, F>
    where
        T: fmt::Debug,
        F: Fn(usize, usize) -> T,
    {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut list = formatter.debug_list();
            for row in 0..self.n {
                list.entry(&(self.get)(row, self.col));
            }
            list.finish()
        }
    }

    let mut list = formatter.debug_list();
    for col in 0..n {
        list.entry(&Column::<T, F> {
            col,
            n,
            get: &get,
            marker: std::marker::PhantomData,
        });
    }
    list.finish()
}
