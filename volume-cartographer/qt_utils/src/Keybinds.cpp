#include "qt_utils/Keybinds.hpp"

namespace qt_utils
{

auto sequenceFor(const ShortcutDef& def) -> QKeySequence
{
    if (def.kind == ShortcutKind::Standard) {
        return QKeySequence(def.standardKey);
    }
    return QKeySequence(QString::fromUtf8(def.sequenceText));
}

auto sequenceFor(const KeyPressDef& def) -> QKeySequence
{
    return QKeySequence(def.key | def.modifiers);
}

auto formatSequence(const QKeySequence& sequence) -> QString
{
    return sequence.toString(QKeySequence::NativeText);
}

// --- KeybindRegistry ---

void KeybindRegistry::registerShortcut(const ShortcutDef& def)
{
    HelpEntry entry{};
    entry.section = def.section;
    entry.description = def.description;
    entry.kind = EntryKind::Shortcut;
    entry.shortcut = &def;
    entries_.append(entry);
}

void KeybindRegistry::registerKeyPress(const KeyPressDef& def)
{
    HelpEntry entry{};
    entry.section = def.section;
    entry.description = def.description;
    entry.kind = EntryKind::KeyPress;
    entry.keypress = &def;
    entries_.append(entry);
}

void KeybindRegistry::registerLiteral(
    const char* section,
    const char* description,
    const char* keyText)
{
    HelpEntry entry{};
    entry.section = section;
    entry.description = description;
    entry.kind = EntryKind::Literal;
    entry.literal = keyText;
    entries_.append(entry);
}

auto KeybindRegistry::findShortcut(const char* id) const -> const ShortcutDef*
{
    for (const auto& entry : entries_) {
        if (entry.kind == EntryKind::Shortcut && entry.shortcut) {
            if (qstrcmp(entry.shortcut->id, id) == 0) {
                return entry.shortcut;
            }
        }
    }
    return nullptr;
}

auto KeybindRegistry::findKeyPress(const char* id) const -> const KeyPressDef*
{
    for (const auto& entry : entries_) {
        if (entry.kind == EntryKind::KeyPress && entry.keypress) {
            if (qstrcmp(entry.keypress->id, id) == 0) {
                return entry.keypress;
            }
        }
    }
    return nullptr;
}

auto KeybindRegistry::buildHelpText() const -> QString
{
    QString result;
    QString currentSection;

    for (const auto& entry : entries_) {
        const QString section = QString::fromUtf8(entry.section);
        if (section != currentSection) {
            if (!result.isEmpty()) {
                result += QStringLiteral("\n");
            }
            result += QStringLiteral("=== ") + section + QStringLiteral(" ===\n");
            currentSection = section;
        }

        QString keyText;
        switch (entry.kind) {
            case EntryKind::Shortcut:
                if (entry.shortcut) {
                    keyText = formatSequence(sequenceFor(*entry.shortcut));
                }
                break;
            case EntryKind::KeyPress:
                if (entry.keypress) {
                    keyText = formatSequence(sequenceFor(*entry.keypress));
                }
                break;
            case EntryKind::Literal:
                if (entry.literal) {
                    keyText = QString::fromUtf8(entry.literal);
                }
                break;
        }

        if (keyText.isEmpty()) {
            continue;
        }

        result +=
            keyText + QStringLiteral(": ") +
            QString::fromUtf8(entry.description) + QStringLiteral("\n");
    }

    return result.trimmed();
}

}  // namespace qt_utils
