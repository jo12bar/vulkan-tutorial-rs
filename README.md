# `vulkan-tutorial-rs`

This is my implementation of the basic renderer described in
[Alexander Overvoorde's `vulkan-tutorial` book][vulkan-tut-orig],
in Rust. I'm primarily following [Kyle Mayes' Rust translation of the tutorial][vulkanalia-tut],
which adds some additional "experimental" chapters about techniques for working
with dynamic scenes. However, there are some large differences between Kyle's
book's code and my code:

1. Instead of using [Kyle Mayes' `vulkanalia` crate][vulkanalia-lib-rs] for
   my Vulkan bindings, I'm using the [`ash` crate][ash-lib-rs], purely because
   most higher-level Rust graphics libraries are also using `ash`. `vulkanalia`
   and `ash` have very similar APIs, as they are both primarily generated
   from the Vulkan API spec, but there are some differences in the helper
   functions implemented in each library.
2. I'm using the [`tracing`][tracing-lib-rs] suite instead of the [`log`][log-lib-rs]
   suite for logging. This gives me span tracking, which I use in many of the
   setup functions via the `#[tracing::instrument]` attribute. [`tracing`]
   actually properly categorizes messages from Vulkan's validation layer into
   the spans, so long as multithreaded processing isn't being used (which
   my code mostly avoids).
3. I'm using [`color-eyre`][color-eyre-lib-rs] instead of [`anyhow`][anyhow-lib-rs]
   because pretty colours sooth my soul when my app crashes (and because
   there's some integration with `tracing`'s spans).
4. To ease the pain on my poor VSCode install (and my poorly-cooled laptop),
   I've split the book's single massive Rust source file into _many_ smaller
   files. This makes rust-analyzer happy.

All commits that follow chapters in Kyle's book are named according to the
chapter title, with one commit per chapter. There are some other commits here
and there for maintenance, refactoring, or bug-fixing.

Once I finish the book, I may begin experimenting with loading GLTF models and
implementing physically-based rendering. Maybe I'll even figure out how skeletal
animation works. Aaaannnd I should probably create proper RAII wrappers for
all the Vulkan objects I use. We'll see :)

[vulkan-tut-orig]: https://vulkan-tutorial.com/
[vulkanalia-tut]: https://kylemayes.github.io/vulkanalia/introduction.html
[vulkanalia-lib-rs]: https://lib.rs/crates/vulkanalia
[ash-lib-rs]: https://lib.rs/crates/ash
[tracing-lib-rs]: https://lib.rs/crates/tracing
[log-lib-rs]: https://lib.rs/crates/log
[color-eyre-lib-rs]: https://lib.rs/crates/color-eyre
[anyhow-lib-rs]: https://lib.rs/crates/anyhow

## License

See [`LICENSE.txt`](./LICENSE.txt).

The [original book][vulkan-tut-orig]'s text
was licensed under the [CC BY-SA 4.0][cc-by-sa-4] license, while its code
listings were licensed under the [CC0 1.0 Universal][cc0-1-universal] license.
Meanwhile, the entirety of [Kyle Mayes' adaptation][vulkanalia-tut] is licensed
under the [Apache 2.0][vulkanalia-license] license. Therefore, as I am not
re-publishing the original book, and using code samples primarily from Kyle
Mayes' adaptation (with reference to the originals), my code is also licensed
under the Apache 2.0 license.

[cc-by-sa-4]: https://creativecommons.org/licenses/by-sa/4.0/
[cc0-1-universal]: https://creativecommons.org/publicdomain/zero/1.0/
