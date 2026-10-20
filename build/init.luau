local process = require("@lune/process")

process.exec("python", {
    "build/work.py",
    "src",
    "src/init/init.luau",
    "-o",
    "out/init.luau"
})