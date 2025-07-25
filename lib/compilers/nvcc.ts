// Copyright (c) 2018, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import * as fs from 'node:fs/promises';
import Path from 'node:path';

import Semver from 'semver';
import _ from 'underscore';

import type {CompilationInfo, CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {PTXAsmParser} from '../parsers/asm-parser-ptx.js';
import {SassAsmParser} from '../parsers/asm-parser-sass.js';
import {asSafeVer} from '../utils.js';

import {ClangParser} from './argument-parsers.js';

export class NvccCompiler extends BaseCompiler {
    static get key() {
        return 'nvcc';
    }

    deviceAsmParser: SassAsmParser;
    ptxParser: PTXAsmParser;

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.compiler.supportsOptOutput = true;
        this.compiler.supportsDeviceAsmView = true;
        this.deviceAsmParser = new SassAsmParser(this.compilerProps);
        this.ptxParser = new PTXAsmParser(this.compilerProps);
    }

    // TODO: (for all of CUDA)
    // * lots of whitespace from nvcc
    // * would be nice to try and filter unused `.func`s from e.g. clang output

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        const opts = ['-o', this.filename(outputFilename), '-g', '-lineinfo', '--keep-device-functions'];
        if (!filters.execute) {
            opts.push('-c', '-keep', '-keep-dir', Path.dirname(outputFilename));
            if (!filters.binary) {
                opts.push('-Xcompiler=-S');
            }
        }
        return opts;
    }

    override getArgumentParserClass() {
        return ClangParser;
    }

    override optOutputRequested(options: string[]) {
        return (
            super.optOutputRequested(options) ||
            options.includes('--optimization-info') ||
            options.includes('-opt-info')
        );
    }

    async nvdisasm(outputFilename: string, result: any, maxOutput: number) {
        const {nvdisasm, semver} = this.compiler;

        const args = Semver.lt(asSafeVer(semver), '11.0.0', true)
            ? [outputFilename, '-c', '-g']
            : [outputFilename, '-c', '-g', '-hex'];

        const {code, execTime, stdout} = await this.exec(unwrap(nvdisasm), args, {
            maxOutput,
            customCwd: result.dirPath,
        });

        if (code === 0) {
            result.objdumpTime = execTime;
            result.asm = this.postProcessObjdumpOutput(stdout);
        } else {
            result.asm = `<No output: ${Path.basename(unwrap(nvdisasm))} returned ${code}>`;
        }
        return result;
    }

    override async postProcess(result, outputFilename: string, filters: ParseFiltersAndOutputOptions) {
        const maxSize = this.env.ceProps('max-asm-size', 64 * 1024 * 1024);
        const optPromise = result.optPath ? this.processOptOutput(result.optPath) : Promise.resolve([]);
        const postProcess = _.compact(this.compiler.postProcess);
        const asmPromise = (
            filters.binary
                ? this.objdump(outputFilename, {}, maxSize, !!filters.intel, !!filters.demangle, false, false, filters)
                : (async () => {
                      if (result.asmSize === undefined) {
                          result.asm = '<No output file>';
                          return result;
                      }
                      if (result.asmSize >= maxSize) {
                          result.asm =
                              '<No output: generated assembly was too large' +
                              ` (${result.asmSize} > ${maxSize} bytes)>`;
                          return result;
                      }
                      if (postProcess.length > 0) {
                          return await this.execPostProcess(result, postProcess, outputFilename, maxSize);
                      }
                      const contents = await fs.readFile(outputFilename, {encoding: 'utf8'});
                      result.asm = contents.toString();
                      return result;
                  })()
        ).then(asm => {
            result.asm = typeof asm === 'string' ? asm : asm.asm;
            return result;
        });
        return Promise.all([asmPromise, optPromise, []]);
    }

    override async extractDeviceCode(
        result: CompilationResult,
        filters: ParseFiltersAndOutputOptions,
        compilationInfo: CompilationInfo,
    ) {
        const {dirPath} = result;
        const {demangle} = filters;
        const devices = {...result.devices};
        if (dirPath) {
            const files = await fs.readdir(dirPath);
            const maxSize = this.env.ceProps('max-asm-size', 64 * 1024 * 1024);
            await Promise.all(
                files
                    .filter(f => f.endsWith('.ptx') || f.endsWith('.cubin'))
                    .map(async name => {
                        const type = name.endsWith('.ptx') ? 'PTX' : 'SASS';
                        const {asm} =
                            type === 'PTX'
                                ? {asm: await fs.readFile(Path.join(dirPath, name), 'utf8')}
                                : await this.nvdisasm(Path.join(dirPath, name), {dirPath}, maxSize);
                        const archAndCode = name.split('.').slice(1, -1).join(', ') || '';
                        const nameAndArch = type + (archAndCode ? ` (${archAndCode.toLowerCase()})` : '');
                        const parser = type === 'PTX' ? this.ptxParser : this.deviceAsmParser;
                        Object.assign(devices, {
                            [nameAndArch]: await this.postProcessAsm(
                                {
                                    okToCache: demangle,
                                    ...parser.process(asm, {...filters, binary: type === 'SASS'}),
                                },
                                {...filters, binary: type === 'SASS'},
                            ),
                        });
                    }),
            );
            result.devices = devices;
        }
        return result;
    }
}
